package xiaomi

import (
	"reflect"
	"testing"
	"unsafe"

	"charm.land/fantasy"
	openaisdk "github.com/charmbracelet/openai-go"
	"github.com/stretchr/testify/require"
)

func TestExtractXiaomiToolCalls(t *testing.T) {
	t.Parallel()

	calls, remain, err := extractXiaomiToolCalls(
		`<function=editor><parameter=command>view</parameter><parameter=file_path>/tmp/report.md</parameter></function>`,
	)
	require.NoError(t, err)
	require.Equal(t, []xiaomiToolCall{{
		name:      "editor",
		arguments: `{"command":"view","file_path":"/tmp/report.md"}`,
	}}, calls)
	require.Empty(t, remain)
}

func TestXiaomiStreamExtraFunc_ParsesXMLToolCalls(t *testing.T) {
	t.Parallel()

	chunk := openaisdk.ChatCompletionChunk{
		Choices: []openaisdk.ChatCompletionChunkChoice{{
			Delta: openaisdk.ChatCompletionChunkChoiceDelta{
				Content: `<function=editor><parameter=command>view</parameter><parameter=file_path>/tmp/report.md</parameter></function>`,
			},
		}},
	}

	var parts []fantasy.StreamPart
	_, ok := xiaomiStreamExtraFunc(chunk, func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	}, map[string]any{})
	require.True(t, ok)
	require.Len(t, parts, 4)
	require.Equal(t, fantasy.StreamPartTypeToolInputStart, parts[0].Type)
	require.Equal(t, "view", parts[0].ToolCallName)
	require.Equal(t, fantasy.StreamPartTypeToolInputDelta, parts[1].Type)
	require.Equal(t, `{"file_path":"/tmp/report.md"}`, parts[1].Delta)
	require.Equal(t, fantasy.StreamPartTypeToolInputEnd, parts[2].Type)
	require.Equal(t, fantasy.StreamPartTypeToolCall, parts[3].Type)
	require.Equal(t, "view", parts[3].ToolCallName)
	require.Equal(t, `{"file_path":"/tmp/report.md"}`, parts[3].ToolCallInput)
}

func TestXiaomiStreamExtraFunc_DegradesMalformedDeltaInsteadOfAborting(t *testing.T) {
	t.Parallel()

	chunk := openaisdk.ChatCompletionChunk{
		Choices: []openaisdk.ChatCompletionChunkChoice{{
			Delta: openaisdk.ChatCompletionChunkChoiceDelta{
				Content: "Continuing after hidden reasoning",
			},
		}},
	}
	forceChunkDeltaRawJSON(t, &chunk.Choices[0].Delta, `{"reasoning_content":"unterminated`)

	var parts []fantasy.StreamPart
	ctx, ok := xiaomiStreamExtraFunc(chunk, func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	}, map[string]any{xiaomiReasoningStartedCtx: true})
	require.True(t, ok)
	require.False(t, ctx[xiaomiReasoningStartedCtx].(bool))
	require.Len(t, parts, 1)
	require.Equal(t, fantasy.StreamPartTypeReasoningEnd, parts[0].Type)
}

func TestXiaomiFilteredStream_SuppressesXMLText(t *testing.T) {
	t.Parallel()

	stream := xiaomiFilteredStream(func(yield func(fantasy.StreamPart) bool) {
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: "0"})
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, ID: "0", Delta: `<function=editor>`})
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: "0"})
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop})
	})

	var parts []fantasy.StreamPart
	stream(func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	})

	require.Len(t, parts, 1)
	require.Equal(t, fantasy.StreamPartTypeFinish, parts[0].Type)
}

func TestNew_ImplementsProvider(t *testing.T) {
	t.Parallel()

	provider, err := New(WithBaseURL("https://token-plan-sgp.xiaomimimo.com/v1"))
	require.NoError(t, err)
	require.Equal(t, Name, provider.Name())
}

func forceChunkDeltaRawJSON(t *testing.T, delta *openaisdk.ChatCompletionChunkChoiceDelta, raw string) {
	t.Helper()

	jsonField := reflect.ValueOf(delta).Elem().FieldByName("JSON")
	rawField := jsonField.FieldByName("raw")
	reflect.NewAt(rawField.Type(), unsafe.Pointer(rawField.UnsafeAddr())).Elem().SetString(raw)
}
