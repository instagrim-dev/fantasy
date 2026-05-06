package bedrockconverse

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Prompt mapping tests ---

func TestMapPrompt_SystemExtracted(t *testing.T) {
	prompt := fantasy.Prompt{
		fantasy.NewSystemMessage("You are a helpful assistant."),
		fantasy.NewUserMessage("Hello"),
	}
	system, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	assert.Len(t, system, 1)
	assert.Len(t, messages, 1)

	// System should be a text block
	sysText, ok := system[0].(*types.SystemContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "You are a helpful assistant.", sysText.Value)

	// Message should be user role
	assert.Equal(t, types.ConversationRoleUser, messages[0].Role)
}

func TestMapPrompt_ToolResultInUserMessage(t *testing.T) {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{
					ToolCallID: "call_123",
					Output:     fantasy.ToolResultOutputContentText{Text: "result text"},
				},
			},
		},
	}
	_, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, types.ConversationRoleUser, messages[0].Role)

	// Content should be a tool result block
	require.Len(t, messages[0].Content, 1)
	tr, ok := messages[0].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok)
	assert.Equal(t, "call_123", aws.ToString(tr.Value.ToolUseId))
}

func TestMapPrompt_ToolErrorResult(t *testing.T) {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{
					ToolCallID: "call_err",
					Output: fantasy.ToolResultOutputContentError{
						Error: assert.AnError,
					},
				},
			},
		},
	}
	_, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	tr, ok := messages[0].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok)
	assert.Equal(t, types.ToolResultStatusError, tr.Value.Status)
}

func TestMapPrompt_AssistantToolCall(t *testing.T) {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID: "tc_1",
					ToolName:   "get_weather",
					Input:      `{"city":"SF"}`,
				},
			},
		},
	}
	_, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, types.ConversationRoleAssistant, messages[0].Role)

	tu, ok := messages[0].Content[0].(*types.ContentBlockMemberToolUse)
	require.True(t, ok)
	assert.Equal(t, "tc_1", aws.ToString(tu.Value.ToolUseId))
	assert.Equal(t, "get_weather", aws.ToString(tu.Value.Name))
}

func TestMapPrompt_ImageInUserMessage(t *testing.T) {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "What is this?"},
				fantasy.FilePart{
					Data:      []byte{0x89, 0x50, 0x4E, 0x47},
					MediaType: "image/png",
				},
			},
		},
	}
	_, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	require.Len(t, messages[0].Content, 2)

	img, ok := messages[0].Content[1].(*types.ContentBlockMemberImage)
	require.True(t, ok)
	assert.Equal(t, types.ImageFormatPng, img.Value.Format)
}

func TestMapPrompt_UnsupportedImageFormat(t *testing.T) {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.FilePart{
					Data:      []byte{0x00},
					MediaType: "image/bmp",
				},
			},
		},
	}
	_, _, err := mapPrompt(prompt)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported image media type")
}

// --- Tool mapping tests ---

func TestMapTools(t *testing.T) {
	tools := []fantasy.Tool{
		fantasy.FunctionTool{
			Name:        "get_weather",
			Description: "Get the weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}

	result := mapTools(tools)
	require.Len(t, result, 1)

	spec, ok := result[0].(*types.ToolMemberToolSpec)
	require.True(t, ok)
	assert.Equal(t, "get_weather", aws.ToString(spec.Value.Name))
	assert.Equal(t, "Get the weather", aws.ToString(spec.Value.Description))
}

func TestMapTools_Empty(t *testing.T) {
	result := mapTools(nil)
	assert.Nil(t, result)
}

// --- ToolChoice mapping tests ---

func TestMapToolChoice_Auto(t *testing.T) {
	tc := fantasy.ToolChoiceAuto
	result := mapToolChoice(&tc)
	_, ok := result.(*types.ToolChoiceMemberAuto)
	assert.True(t, ok)
}

func TestMapToolChoice_Required(t *testing.T) {
	tc := fantasy.ToolChoiceRequired
	result := mapToolChoice(&tc)
	_, ok := result.(*types.ToolChoiceMemberAny)
	assert.True(t, ok)
}

func TestMapToolChoice_None(t *testing.T) {
	tc := fantasy.ToolChoiceNone
	result := mapToolChoice(&tc)
	assert.Nil(t, result)
}

func TestMapToolChoice_Specific(t *testing.T) {
	tc := fantasy.SpecificToolChoice("get_weather")
	result := mapToolChoice(&tc)
	specific, ok := result.(*types.ToolChoiceMemberTool)
	require.True(t, ok)
	assert.Equal(t, "get_weather", aws.ToString(specific.Value.Name))
}

func TestMapToolChoice_Nil(t *testing.T) {
	result := mapToolChoice(nil)
	assert.Nil(t, result)
}

// --- InferenceConfig mapping tests ---

func TestMapInferenceConfig(t *testing.T) {
	maxTokens := int64(1024)
	temp := 0.7
	topP := 0.9
	call := fantasy.Call{
		MaxOutputTokens: &maxTokens,
		Temperature:     &temp,
		TopP:            &topP,
	}

	ic := mapInferenceConfig(call)
	require.NotNil(t, ic)
	assert.Equal(t, int32(1024), *ic.MaxTokens)
	assert.InDelta(t, float32(0.7), *ic.Temperature, 0.001)
	assert.InDelta(t, float32(0.9), *ic.TopP, 0.001)
}

func TestMapInferenceConfig_Empty(t *testing.T) {
	ic := mapInferenceConfig(fantasy.Call{})
	assert.Nil(t, ic)
}

// --- Response mapping tests ---

func TestMapResponse_TextResponse(t *testing.T) {
	out := &bedrockruntime.ConverseOutput{
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:  aws.Int32(10),
			OutputTokens: aws.Int32(20),
			TotalTokens:  aws.Int32(30),
		},
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "Hello there!"},
				},
			},
		},
	}

	resp := mapResponse(out)
	assert.Equal(t, fantasy.FinishReasonStop, resp.FinishReason)
	assert.Equal(t, int64(10), resp.Usage.InputTokens)
	assert.Equal(t, int64(20), resp.Usage.OutputTokens)
	assert.Equal(t, int64(30), resp.Usage.TotalTokens)
	assert.Equal(t, "Hello there!", resp.Content.Text())
}

func TestMapResponse_ToolCallResponse(t *testing.T) {
	out := &bedrockruntime.ConverseOutput{
		StopReason: types.StopReasonToolUse,
		Usage: &types.TokenUsage{
			InputTokens:  aws.Int32(5),
			OutputTokens: aws.Int32(15),
			TotalTokens:  aws.Int32(20),
		},
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String("call_abc"),
							Name:      aws.String("get_weather"),
							Input:     nil,
						},
					},
				},
			},
		},
	}

	resp := mapResponse(out)
	assert.Equal(t, fantasy.FinishReasonToolCalls, resp.FinishReason)

	toolCalls := resp.Content.ToolCalls()
	require.Len(t, toolCalls, 1)
	assert.Equal(t, "call_abc", toolCalls[0].ToolCallID)
	assert.Equal(t, "get_weather", toolCalls[0].ToolName)
}

// --- StopReason mapping tests ---

func TestMapStopReason(t *testing.T) {
	tests := []struct {
		input    types.StopReason
		expected fantasy.FinishReason
	}{
		{types.StopReasonEndTurn, fantasy.FinishReasonStop},
		{types.StopReasonToolUse, fantasy.FinishReasonToolCalls},
		{types.StopReasonMaxTokens, fantasy.FinishReasonLength},
		{types.StopReasonContentFiltered, fantasy.FinishReasonContentFilter},
		{types.StopReason("something_else"), fantasy.FinishReasonUnknown},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.expected, mapStopReason(tt.input), "stop reason: %s", tt.input)
	}
}

// --- Stream event mapping tests ---

// mockReader implements bedrockruntime.ConverseStreamOutputReader for testing.
type mockReader struct {
	events []types.ConverseStreamOutput
	ch     chan types.ConverseStreamOutput
	err    error
}

func newMockReader(events []types.ConverseStreamOutput) *mockReader {
	ch := make(chan types.ConverseStreamOutput, len(events))
	for _, e := range events {
		ch <- e
	}
	close(ch)
	return &mockReader{events: events, ch: ch}
}

func (m *mockReader) Events() <-chan types.ConverseStreamOutput { return m.ch }
func (m *mockReader) Close() error                              { return nil }
func (m *mockReader) Err() error                                { return m.err }


func TestStreamEvent_TextOnly(t *testing.T) {
	state := &streamState{}

	var parts []fantasy.StreamPart

	// Content block start (text) — no explicit text start in Converse
	events := []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{
				ContentBlockIndex: aws.Int32(0),
				// No Start field for text blocks (only ToolUse has a start member)
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta:             &types.ContentBlockDeltaMemberText{Value: "Hello"},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta:             &types.ContentBlockDeltaMemberText{Value: " world"},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockStop{
			Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)},
		},
		&types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn},
		},
		&types.ConverseStreamOutputMemberMetadata{
			Value: types.ConverseStreamMetadataEvent{
				Usage: &types.TokenUsage{
					InputTokens:  aws.Int32(5),
					OutputTokens: aws.Int32(2),
					TotalTokens:  aws.Int32(7),
				},
			},
		},
	}

	for _, event := range events {
		parts = append(parts, mapStreamEvent(event, state)...)
	}

	require.Len(t, parts, 5)
	assert.Equal(t, fantasy.StreamPartTypeTextStart, parts[0].Type)
	assert.Equal(t, fantasy.StreamPartTypeTextDelta, parts[1].Type)
	assert.Equal(t, "Hello", parts[1].Delta)
	assert.Equal(t, fantasy.StreamPartTypeTextDelta, parts[2].Type)
	assert.Equal(t, " world", parts[2].Delta)
	assert.Equal(t, fantasy.StreamPartTypeTextEnd, parts[3].Type)
	assert.Equal(t, fantasy.StreamPartTypeFinish, parts[4].Type)
	assert.Equal(t, fantasy.FinishReasonStop, parts[4].FinishReason)
	assert.Equal(t, int64(5), parts[4].Usage.InputTokens)
}

func TestStreamEvent_ToolCall(t *testing.T) {
	state := &streamState{}
	var parts []fantasy.StreamPart

	events := []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{
				ContentBlockIndex: aws.Int32(0),
				Start: &types.ContentBlockStartMemberToolUse{
					Value: types.ToolUseBlockStart{
						ToolUseId: aws.String("call_1"),
						Name:      aws.String("get_weather"),
					},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberToolUse{
					Value: types.ToolUseBlockDelta{Input: aws.String(`{"city":`)},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberToolUse{
					Value: types.ToolUseBlockDelta{Input: aws.String(`"SF"}`)},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockStop{
			Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)},
		},
		&types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse},
		},
		&types.ConverseStreamOutputMemberMetadata{
			Value: types.ConverseStreamMetadataEvent{
				Usage: &types.TokenUsage{
					InputTokens:  aws.Int32(10),
					OutputTokens: aws.Int32(5),
					TotalTokens:  aws.Int32(15),
				},
			},
		},
	}

	for _, event := range events {
		parts = append(parts, mapStreamEvent(event, state)...)
	}

	require.Len(t, parts, 6)
	assert.Equal(t, fantasy.StreamPartTypeToolInputStart, parts[0].Type)
	assert.Equal(t, "call_1", parts[0].ID)
	assert.Equal(t, "get_weather", parts[0].ToolCallName)

	assert.Equal(t, fantasy.StreamPartTypeToolInputDelta, parts[1].Type)
	assert.Equal(t, `{"city":`, parts[1].Delta)

	assert.Equal(t, fantasy.StreamPartTypeToolInputDelta, parts[2].Type)
	assert.Equal(t, `"SF"}`, parts[2].Delta)

	assert.Equal(t, fantasy.StreamPartTypeToolInputEnd, parts[3].Type)

	assert.Equal(t, fantasy.StreamPartTypeToolCall, parts[4].Type)
	assert.Equal(t, "call_1", parts[4].ID)
	assert.Equal(t, "get_weather", parts[4].ToolCallName)
	assert.Equal(t, `{"city":"SF"}`, parts[4].ToolCallInput)

	assert.Equal(t, fantasy.StreamPartTypeFinish, parts[5].Type)
	assert.Equal(t, fantasy.FinishReasonToolCalls, parts[5].FinishReason)
}

func TestStreamEvent_MixedTextAndToolCall(t *testing.T) {
	state := &streamState{}
	var parts []fantasy.StreamPart

	events := []types.ConverseStreamOutput{
		// Text block
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0)},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta:             &types.ContentBlockDeltaMemberText{Value: "Let me check."},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockStop{
			Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)},
		},
		// Tool block
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{
				ContentBlockIndex: aws.Int32(1),
				Start: &types.ContentBlockStartMemberToolUse{
					Value: types.ToolUseBlockStart{
						ToolUseId: aws.String("tc_2"),
						Name:      aws.String("search"),
					},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(1),
				Delta: &types.ContentBlockDeltaMemberToolUse{
					Value: types.ToolUseBlockDelta{Input: aws.String(`{"q":"test"}`)},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockStop{
			Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(1)},
		},
		&types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse},
		},
		&types.ConverseStreamOutputMemberMetadata{
			Value: types.ConverseStreamMetadataEvent{
				Usage: &types.TokenUsage{
					InputTokens:  aws.Int32(20),
					OutputTokens: aws.Int32(10),
					TotalTokens:  aws.Int32(30),
				},
			},
		},
	}

	for _, event := range events {
		parts = append(parts, mapStreamEvent(event, state)...)
	}

	// TextStart, TextDelta, TextEnd, ToolInputStart, ToolInputDelta, ToolInputEnd, ToolCall, Finish
	require.Len(t, parts, 8)
	assert.Equal(t, fantasy.StreamPartTypeTextStart, parts[0].Type)
	assert.Equal(t, fantasy.StreamPartTypeTextDelta, parts[1].Type)
	assert.Equal(t, "Let me check.", parts[1].Delta)
	assert.Equal(t, fantasy.StreamPartTypeTextEnd, parts[2].Type)
	assert.Equal(t, fantasy.StreamPartTypeToolInputStart, parts[3].Type)
	assert.Equal(t, fantasy.StreamPartTypeToolInputDelta, parts[4].Type)
	assert.Equal(t, fantasy.StreamPartTypeToolInputEnd, parts[5].Type)
	assert.Equal(t, fantasy.StreamPartTypeToolCall, parts[6].Type)
	assert.Equal(t, `{"q":"test"}`, parts[6].ToolCallInput)
	assert.Equal(t, fantasy.StreamPartTypeFinish, parts[7].Type)
}

// --- Error mapping tests ---

func TestWrapAWSError_Nil(t *testing.T) {
	assert.Nil(t, wrapAWSError(nil))
}

func TestWrapAWSError_Generic(t *testing.T) {
	err := wrapAWSError(assert.AnError)
	pe, ok := err.(*fantasy.ProviderError)
	require.True(t, ok)
	assert.Equal(t, 500, pe.StatusCode)
}

// throttlingError is a mock AWS error for testing.
type throttlingError struct{}

func (throttlingError) Error() string        { return "Rate exceeded" }
func (throttlingError) ErrorCode() string    { return "ThrottlingException" }
func (throttlingError) ErrorMessage() string { return "Rate exceeded" }

func TestWrapAWSError_Throttling(t *testing.T) {
	err := wrapAWSError(throttlingError{})
	pe, ok := err.(*fantasy.ProviderError)
	require.True(t, ok)
	assert.Equal(t, 429, pe.StatusCode)
	assert.True(t, pe.IsRetryable())
}

// --- Provider tests ---

func TestProviderName(t *testing.T) {
	p, err := New()
	require.NoError(t, err)
	assert.Equal(t, Name, p.Name())
}

func TestProviderName_Custom(t *testing.T) {
	p, err := New(WithName("custom"))
	require.NoError(t, err)
	assert.Equal(t, "custom", p.Name())
}

// --- jsonToMap tests ---

func TestJsonToMap(t *testing.T) {
	result := jsonToMap(`{"key":"value"}`)
	assert.Equal(t, "value", result["key"])
}

func TestJsonToMap_Empty(t *testing.T) {
	assert.Nil(t, jsonToMap(""))
}

func TestJsonToMap_Invalid(t *testing.T) {
	assert.Nil(t, jsonToMap("not json"))
}

// --- Media type mapping tests ---

func TestMediaTypeToImageFormat(t *testing.T) {
	tests := []struct {
		mediaType string
		expected  types.ImageFormat
		wantErr   bool
	}{
		{"image/png", types.ImageFormatPng, false},
		{"image/jpeg", types.ImageFormatJpeg, false},
		{"image/gif", types.ImageFormatGif, false},
		{"image/webp", types.ImageFormatWebp, false},
		{"image/bmp", "", true},
	}
	for _, tt := range tests {
		format, err := mediaTypeToImageFormat(tt.mediaType)
		if tt.wantErr {
			assert.Error(t, err)
		} else {
			require.NoError(t, err)
			assert.Equal(t, tt.expected, format)
		}
	}
}

// --- Response content mapping test with tool input decoding ---

func TestMapResponseContent_ToolUseWithNilInput(t *testing.T) {
	blocks := []types.ContentBlock{
		&types.ContentBlockMemberToolUse{
			Value: types.ToolUseBlock{
				ToolUseId: aws.String("tc_nil"),
				Name:      aws.String("noop"),
				Input:     nil,
			},
		},
	}
	content := mapResponseContent(blocks)
	require.Len(t, content, 1)
	tc, ok := fantasy.AsContentType[fantasy.ToolCallContent](content[0])
	require.True(t, ok)
	assert.Equal(t, "{}", tc.Input) // defaults to empty object
}

// --- Full round-trip: prompt with tool result containing JSON ---

func TestToolResultPart_JSONRoundTrip(t *testing.T) {
	resultData := map[string]any{"temp": 72, "unit": "F"}
	resultJSON, _ := json.Marshal(resultData)

	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{
					ToolCallID: "call_weather",
					Output:     fantasy.ToolResultOutputContentText{Text: string(resultJSON)},
				},
			},
		},
	}

	_, messages, err := mapPrompt(prompt)
	require.NoError(t, err)
	require.Len(t, messages, 1)

	tr, ok := messages[0].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok)
	assert.Equal(t, "call_weather", aws.ToString(tr.Value.ToolUseId))

	textBlock, ok := tr.Value.Content[0].(*types.ToolResultContentBlockMemberText)
	require.True(t, ok)
	assert.Contains(t, textBlock.Value, "temp")
}
