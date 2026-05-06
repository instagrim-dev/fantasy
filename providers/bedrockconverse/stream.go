package bedrockconverse

import (
	"strings"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// blockKind tracks what type of content block is currently active in the stream.
type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockToolUse
)

// streamState tracks accumulated state across stream events.
type streamState struct {
	activeBlock   blockKind
	toolID        string
	toolName      string
	toolInput     strings.Builder
	textStarted   bool
	usage         fantasy.Usage
	finishReason  fantasy.FinishReason
}

// newStreamResponse converts a Converse stream into a fantasy StreamResponse.
func newStreamResponse(out *bedrockruntime.ConverseStreamOutput) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		stream := out.GetStream()
		defer stream.Close()

		var state streamState

		for event := range stream.Events() {
			parts := mapStreamEvent(event, &state)
			for _, part := range parts {
				if !yield(part) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: wrapAWSError(err),
			})
		}
	}
}

// mapStreamEvent converts a single Converse stream event into zero or more StreamParts.
func mapStreamEvent(event types.ConverseStreamOutput, state *streamState) []fantasy.StreamPart {
	switch e := event.(type) {
	case *types.ConverseStreamOutputMemberContentBlockStart:
		return handleContentBlockStart(e.Value, state)

	case *types.ConverseStreamOutputMemberContentBlockDelta:
		return handleContentBlockDelta(e.Value, state)

	case *types.ConverseStreamOutputMemberContentBlockStop:
		return handleContentBlockStop(state)

	case *types.ConverseStreamOutputMemberMessageStop:
		state.finishReason = mapStopReason(e.Value.StopReason)
		return nil // finish emitted after metadata

	case *types.ConverseStreamOutputMemberMetadata:
		if e.Value.Usage != nil {
			state.usage = mapUsage(e.Value.Usage)
		}
		// Metadata is the last event; emit finish.
		return []fantasy.StreamPart{{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: state.finishReason,
			Usage:        state.usage,
		}}

	default:
		return nil
	}
}

func handleContentBlockStart(e types.ContentBlockStartEvent, state *streamState) []fantasy.StreamPart {
	switch start := e.Start.(type) {
	case *types.ContentBlockStartMemberToolUse:
		state.activeBlock = blockToolUse
		state.toolID = aws.ToString(start.Value.ToolUseId)
		state.toolName = aws.ToString(start.Value.Name)
		state.toolInput.Reset()
		return []fantasy.StreamPart{{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           state.toolID,
			ToolCallName: state.toolName,
		}}
	default:
		// Text or other block; we emit TextStart on the first delta.
		state.activeBlock = blockText
		state.textStarted = false
		return nil
	}
}

func handleContentBlockDelta(e types.ContentBlockDeltaEvent, state *streamState) []fantasy.StreamPart {
	switch d := e.Delta.(type) {
	case *types.ContentBlockDeltaMemberText:
		var parts []fantasy.StreamPart
		if !state.textStarted {
			parts = append(parts, fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart})
			state.textStarted = true
			state.activeBlock = blockText
		}
		parts = append(parts, fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			Delta: d.Value,
		})
		return parts

	case *types.ContentBlockDeltaMemberToolUse:
		input := aws.ToString(d.Value.Input)
		state.toolInput.WriteString(input)
		return []fantasy.StreamPart{{
			Type:  fantasy.StreamPartTypeToolInputDelta,
			Delta: input,
		}}

	default:
		return nil
	}
}

func handleContentBlockStop(state *streamState) []fantasy.StreamPart {
	var parts []fantasy.StreamPart

	switch state.activeBlock {
	case blockText:
		if state.textStarted {
			parts = append(parts, fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd})
		}
	case blockToolUse:
		parts = append(parts, fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
		})
		parts = append(parts, fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolCall,
			ID:            state.toolID,
			ToolCallName:  state.toolName,
			ToolCallInput: state.toolInput.String(),
		})
	}

	state.activeBlock = blockNone
	state.textStarted = false
	return parts
}
