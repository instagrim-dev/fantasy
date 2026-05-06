package bedrockconverse

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// mapPrompt converts a fantasy Prompt into Converse system blocks and messages.
// System messages are extracted into a separate slice (Converse puts them in a
// top-level field, not in the message array). Tool-role messages are folded into
// user messages with ToolResult content blocks (Converse has no tool role).
func mapPrompt(prompt fantasy.Prompt) ([]types.SystemContentBlock, []types.Message, error) {
	var system []types.SystemContentBlock
	var messages []types.Message

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			for _, part := range msg.Content {
				if tp, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
					system = append(system, &types.SystemContentBlockMemberText{
						Value: tp.Text,
					})
				}
			}

		case fantasy.MessageRoleUser:
			blocks, err := mapUserContentBlocks(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: blocks,
			})

		case fantasy.MessageRoleAssistant:
			blocks := mapAssistantContentBlocks(msg.Content)
			messages = append(messages, types.Message{
				Role:    types.ConversationRoleAssistant,
				Content: blocks,
			})

		case fantasy.MessageRoleTool:
			// Converse has no tool role. Tool results go in a user message.
			blocks, err := mapToolResultBlocks(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: blocks,
			})
		}
	}

	return system, messages, nil
}

func mapUserContentBlocks(parts []fantasy.MessagePart) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock
	for _, part := range parts {
		switch p := part.(type) {
		case fantasy.TextPart:
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		case fantasy.FilePart:
			imgBlock, err := mapFilePart(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, imgBlock)
		case fantasy.ToolResultPart:
			trBlock, err := mapToolResultPart(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, &types.ContentBlockMemberToolResult{Value: trBlock})
		}
	}
	return blocks, nil
}

func mapAssistantContentBlocks(parts []fantasy.MessagePart) []types.ContentBlock {
	var blocks []types.ContentBlock
	for _, part := range parts {
		switch p := part.(type) {
		case fantasy.TextPart:
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		case fantasy.ToolCallPart:
			blocks = append(blocks, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: aws.String(p.ToolCallID),
					Name:      aws.String(p.ToolName),
					Input:     document.NewLazyDocument(jsonToMap(p.Input)),
				},
			})
		}
	}
	return blocks
}

func mapToolResultBlocks(parts []fantasy.MessagePart) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock
	for _, part := range parts {
		if trp, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part); ok {
			trBlock, err := mapToolResultPart(trp)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, &types.ContentBlockMemberToolResult{Value: trBlock})
		}
	}
	return blocks, nil
}

func mapToolResultPart(p fantasy.ToolResultPart) (types.ToolResultBlock, error) {
	block := types.ToolResultBlock{
		ToolUseId: aws.String(p.ToolCallID),
	}

	if p.Output == nil {
		return block, nil
	}

	switch out := p.Output.(type) {
	case fantasy.ToolResultOutputContentText:
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberText{Value: out.Text},
		}
	case fantasy.ToolResultOutputContentError:
		block.Status = types.ToolResultStatusError
		errMsg := ""
		if out.Error != nil {
			errMsg = out.Error.Error()
		}
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberText{Value: errMsg},
		}
	case fantasy.ToolResultOutputContentMedia:
		// Send media as JSON document
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberJson{
				Value: document.NewLazyDocument(map[string]any{
					"type":       "media",
					"media_type": out.MediaType,
					"data":       out.Data,
				}),
			},
		}
	default:
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberText{Value: fmt.Sprintf("%v", out)},
		}
	}

	return block, nil
}

func mapFilePart(p fantasy.FilePart) (types.ContentBlock, error) {
	format, err := mediaTypeToImageFormat(p.MediaType)
	if err != nil {
		return nil, err
	}
	return &types.ContentBlockMemberImage{
		Value: types.ImageBlock{
			Format: format,
			Source: &types.ImageSourceMemberBytes{Value: p.Data},
		},
	}, nil
}

func mediaTypeToImageFormat(mediaType string) (types.ImageFormat, error) {
	switch strings.ToLower(mediaType) {
	case "image/png":
		return types.ImageFormatPng, nil
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg, nil
	case "image/gif":
		return types.ImageFormatGif, nil
	case "image/webp":
		return types.ImageFormatWebp, nil
	default:
		return "", fmt.Errorf("bedrockconverse: unsupported image media type: %s", mediaType)
	}
}

// mapTools converts fantasy tools to Converse tool specifications.
func mapTools(tools []fantasy.Tool) []types.Tool {
	if len(tools) == 0 {
		return nil
	}
	var result []types.Tool
	for _, t := range tools {
		ft, ok := t.(fantasy.FunctionTool)
		if !ok {
			continue
		}
		spec := types.ToolSpecification{
			Name:        aws.String(ft.Name),
			Description: aws.String(ft.Description),
			InputSchema: &types.ToolInputSchemaMemberJson{
				Value: document.NewLazyDocument(ft.InputSchema),
			},
		}
		result = append(result, &types.ToolMemberToolSpec{Value: spec})
	}
	return result
}

// mapToolChoice converts a fantasy ToolChoice to the Converse ToolChoice union.
func mapToolChoice(choice *fantasy.ToolChoice) types.ToolChoice {
	if choice == nil {
		return nil
	}
	switch *choice {
	case fantasy.ToolChoiceAuto:
		return &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
	case fantasy.ToolChoiceRequired:
		return &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
	case fantasy.ToolChoiceNone:
		return nil // no tool choice constraint
	default:
		// Specific tool name
		return &types.ToolChoiceMemberTool{
			Value: types.SpecificToolChoice{
				Name: aws.String(string(*choice)),
			},
		}
	}
}

// mapInferenceConfig builds the Converse InferenceConfiguration from a Call.
func mapInferenceConfig(call fantasy.Call) *types.InferenceConfiguration {
	var ic types.InferenceConfiguration
	var set bool

	if call.MaxOutputTokens != nil {
		v := int32(*call.MaxOutputTokens)
		ic.MaxTokens = &v
		set = true
	}
	if call.Temperature != nil {
		v := float32(*call.Temperature)
		ic.Temperature = &v
		set = true
	}
	if call.TopP != nil {
		v := float32(*call.TopP)
		ic.TopP = &v
		set = true
	}

	if !set {
		return nil
	}
	return &ic
}

// mapResponse converts a Converse response into a fantasy Response.
func mapResponse(out *bedrockruntime.ConverseOutput) *fantasy.Response {
	resp := &fantasy.Response{
		FinishReason: mapStopReason(out.StopReason),
	}

	if out.Usage != nil {
		resp.Usage = mapUsage(out.Usage)
	}

	if msgOut, ok := out.Output.(*types.ConverseOutputMemberMessage); ok {
		resp.Content = mapResponseContent(msgOut.Value.Content)
	}

	return resp
}

func mapResponseContent(blocks []types.ContentBlock) fantasy.ResponseContent {
	var content fantasy.ResponseContent
	for _, block := range blocks {
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			content = append(content, fantasy.TextContent{Text: b.Value})
		case *types.ContentBlockMemberToolUse:
			inputJSON := "{}"
			if b.Value.Input != nil {
				var raw map[string]any
				if err := b.Value.Input.UnmarshalSmithyDocument(&raw); err == nil {
					if data, err := json.Marshal(raw); err == nil {
						inputJSON = string(data)
					}
				}
			}
			content = append(content, fantasy.ToolCallContent{
				ToolCallID: aws.ToString(b.Value.ToolUseId),
				ToolName:   aws.ToString(b.Value.Name),
				Input:      inputJSON,
			})
		}
	}
	return content
}

func mapStopReason(reason types.StopReason) fantasy.FinishReason {
	switch reason {
	case types.StopReasonEndTurn:
		return fantasy.FinishReasonStop
	case types.StopReasonToolUse:
		return fantasy.FinishReasonToolCalls
	case types.StopReasonMaxTokens:
		return fantasy.FinishReasonLength
	case types.StopReasonContentFiltered:
		return fantasy.FinishReasonContentFilter
	default:
		return fantasy.FinishReasonUnknown
	}
}

func mapUsage(u *types.TokenUsage) fantasy.Usage {
	var usage fantasy.Usage
	if u.InputTokens != nil {
		usage.InputTokens = int64(*u.InputTokens)
	}
	if u.OutputTokens != nil {
		usage.OutputTokens = int64(*u.OutputTokens)
	}
	if u.TotalTokens != nil {
		usage.TotalTokens = int64(*u.TotalTokens)
	}
	return usage
}

// jsonToMap parses a JSON string into a map. Returns nil on error or empty input.
func jsonToMap(s string) map[string]any {
	if s == "" {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil
	}
	return m
}
