// Package xiaomi provides a fantasy.Provider for Xiaomi-compatible OpenAI APIs.
package xiaomi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	openaisdk "github.com/charmbracelet/openai-go"
)

const (
	// Name is the provider type name for Xiaomi.
	Name = "xiaomi"
	// DefaultURL is Xiaomi's standard OpenAI-compatible endpoint.
	DefaultURL = "https://api.xiaomimimo.com/v1"
)

const (
	xiaomiReasoningStartedCtx = "xiaomi_reasoning_started"
	xiaomiToolCallBufferCtx   = "xiaomi_tool_call_buffer"
)

type options struct {
	baseURL    string
	apiKey     string
	headers    map[string]string
	httpClient *http.Client
	extraBody  map[string]any
	thinking   bool
}

// Option configures the Xiaomi provider.
type Option = func(*options)

type provider struct {
	inner fantasy.Provider
	opts  options
}

type languageModel struct {
	fantasy.LanguageModel
}

type xiaomiToolCall struct {
	name      string
	arguments string
}

// WithBaseURL sets the base URL for the Xiaomi provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithAPIKey sets the API key for the Xiaomi provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithHeaders sets the headers for the Xiaomi provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.headers = headers
	}
}

// WithHTTPClient sets the HTTP client for the Xiaomi provider.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(o *options) {
		o.httpClient = httpClient
	}
}

// WithExtraBody sets additional request-body fields for the Xiaomi provider.
func WithExtraBody(extraBody map[string]any) Option {
	return func(o *options) {
		o.extraBody = extraBody
	}
}

// WithThinking enables or disables Xiaomi's thinking mode request field.
func WithThinking(enabled bool) Option {
	return func(o *options) {
		o.thinking = enabled
	}
}

// New creates a new Xiaomi provider.
func New(opts ...Option) (fantasy.Provider, error) {
	o := options{
		baseURL:   DefaultURL,
		headers:   map[string]string{},
		extraBody: map[string]any{},
	}
	for _, opt := range opts {
		opt(&o)
	}

	openaiCompatOpts := []openaicompat.Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(o.baseURL),
		openaicompat.WithAPIKey(o.apiKey),
		openaicompat.WithLanguageModelOption(
			openai.WithLanguageModelPrepareCallFunc(xiaomiPrepareCallFunc(&o)),
		),
		openaicompat.WithLanguageModelOption(
			openai.WithLanguageModelStreamExtraFunc(xiaomiStreamExtraFunc),
		),
	}

	if len(o.headers) > 0 {
		openaiCompatOpts = append(openaiCompatOpts, openaicompat.WithHeaders(o.headers))
	}
	if o.httpClient != nil {
		openaiCompatOpts = append(openaiCompatOpts, openaicompat.WithHTTPClient(o.httpClient))
	}

	inner, err := openaicompat.New(openaiCompatOpts...)
	if err != nil {
		return nil, err
	}
	return &provider{inner: inner, opts: o}, nil
}

func (p *provider) Name() string {
	return Name
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	lm, err := p.inner.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, err
	}
	return &languageModel{LanguageModel: lm}, nil
}

func (m *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	stream, err := m.LanguageModel.Stream(ctx, call)
	if err != nil {
		return nil, err
	}
	return xiaomiFilteredStream(stream), nil
}

func xiaomiPrepareCallFunc(opts *options) openai.LanguageModelPrepareCallFunc {
	return func(model fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
		warnings, err := openaicompat.PrepareCallFunc(model, params, call)
		if err != nil {
			return warnings, err
		}

		extraFields := map[string]any{}
		if opts.thinking {
			extraFields["thinking"] = map[string]any{"type": "enabled"}
		}
		for k, v := range opts.extraBody {
			extraFields[k] = v
		}
		if len(extraFields) > 0 {
			params.SetExtraFields(extraFields)
		}
		return warnings, nil
	}
}

func xiaomiStreamExtraFunc(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}
	if ctx == nil {
		ctx = map[string]any{}
	}

	reasoningStarted, _ := ctx[xiaomiReasoningStartedCtx].(bool)

	for inx, choice := range chunk.Choices {
		choiceID := strconv.Itoa(inx)
		emitReasoningEnd := func() bool {
			if reasoningStarted && (choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0) {
				reasoningStarted = false
				ctx[xiaomiReasoningStartedCtx] = false
				return yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningEnd,
					ID:   choiceID,
				})
			}
			return true
		}

		if choice.Delta.Content != "" || hasBufferedToolCallContent(ctx) {
			var ok bool
			ctx, ok = parseXiaomiToolCalls(choice.Delta.Content, yield, ctx, choiceID)
			if !ok {
				return ctx, false
			}
		}

		raw := strings.TrimSpace(choice.Delta.RawJSON())
		if raw == "" {
			if !emitReasoningEnd() {
				return ctx, false
			}
			continue
		}

		var reasoning struct {
			ReasoningContent string `json:"reasoning_content"`
		}
		if err := json.Unmarshal([]byte(raw), &reasoning); err != nil {
			if !emitReasoningEnd() {
				return ctx, false
			}
			continue
		}

		if reasoning.ReasoningContent != "" {
			if !reasoningStarted {
				reasoningStarted = true
				ctx[xiaomiReasoningStartedCtx] = true
				if !yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningStart,
					ID:   choiceID,
				}) {
					return ctx, false
				}
			}
			if !yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeReasoningDelta,
				ID:    choiceID,
				Delta: reasoning.ReasoningContent,
			}) {
				return ctx, false
			}
			continue
		}

		if !emitReasoningEnd() {
			return ctx, false
		}
	}

	return ctx, true
}

func xiaomiFilteredStream(original fantasy.StreamResponse) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		var pendingTextStart *fantasy.StreamPart
		suppressXMLText := false

		original(func(part fantasy.StreamPart) bool {
			switch part.Type {
			case fantasy.StreamPartTypeTextStart:
				cp := part
				pendingTextStart = &cp
				return true
			case fantasy.StreamPartTypeTextDelta:
				if suppressXMLText || strings.Contains(part.Delta, "<function=") {
					suppressXMLText = true
					return true
				}
				if pendingTextStart != nil {
					if !yield(*pendingTextStart) {
						return false
					}
					pendingTextStart = nil
				}
				return yield(part)
			case fantasy.StreamPartTypeTextEnd:
				if suppressXMLText {
					suppressXMLText = false
					pendingTextStart = nil
					return true
				}
				if pendingTextStart != nil {
					pendingTextStart = nil
					return true
				}
				return yield(part)
			case fantasy.StreamPartTypeFinish, fantasy.StreamPartTypeError:
				suppressXMLText = false
				pendingTextStart = nil
			}
			return yield(part)
		})
	}
}

func hasBufferedToolCallContent(ctx map[string]any) bool {
	buffer, _ := ctx[xiaomiToolCallBufferCtx].(string)
	return buffer != ""
}

func parseXiaomiToolCalls(fragment string, yield func(fantasy.StreamPart) bool, ctx map[string]any, choiceID string) (map[string]any, bool) {
	buffer, _ := ctx[xiaomiToolCallBufferCtx].(string)
	buffer += fragment

	toolCalls, remaining, err := extractXiaomiToolCalls(buffer)
	if err != nil {
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: &fantasy.Error{Title: "parse error", Message: "error parsing Xiaomi tool calls", Cause: err},
		})
		return ctx, false
	}
	ctx[xiaomiToolCallBufferCtx] = remaining

	for i, tc := range toolCalls {
		toolName := tc.name
		toolArgs := tc.arguments
		if isToolWrapperName(toolName) {
			var argsMap map[string]string
			if err := json.Unmarshal([]byte(tc.arguments), &argsMap); err == nil {
				if command := strings.TrimSpace(argsMap["command"]); command != "" {
					toolName = command
					delete(argsMap, "command")
					if normalized, err := json.Marshal(argsMap); err == nil {
						toolArgs = string(normalized)
					}
				}
			}
		}

		toolCallID := fmt.Sprintf("xiaomi_%s_%d", choiceID, i)
		if !yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           toolCallID,
			ToolCallName: toolName,
		}) {
			return ctx, false
		}
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeToolInputDelta,
			ID:    toolCallID,
			Delta: toolArgs,
		}) {
			return ctx, false
		}
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   toolCallID,
		}) {
			return ctx, false
		}
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolCall,
			ID:            toolCallID,
			ToolCallName:  toolName,
			ToolCallInput: toolArgs,
		}) {
			return ctx, false
		}
	}

	return ctx, true
}

func extractXiaomiToolCalls(content string) ([]xiaomiToolCall, string, error) {
	var toolCalls []xiaomiToolCall

	funcPattern := regexp.MustCompile(`(?s)<function=([^>]+)>(.*?)</function>`)
	paramsPattern := regexp.MustCompile(`<parameter=([^>]+)>(.*?)</parameter>`)
	matches := funcPattern.FindAllStringSubmatchIndex(content, -1)
	if len(matches) == 0 {
		return toolCalls, content, nil
	}

	for _, match := range matches {
		if len(match) < 6 || match[2] == -1 || match[4] == -1 {
			continue
		}

		funcName := strings.TrimSpace(content[match[2]:match[3]])
		funcBody := content[match[4]:match[5]]

		params := map[string]string{}
		for _, pm := range paramsPattern.FindAllStringSubmatch(funcBody, -1) {
			if len(pm) != 3 {
				continue
			}
			params[strings.TrimSpace(pm[1])] = strings.TrimSpace(pm[2])
		}

		argsJSON, err := json.Marshal(params)
		if err != nil {
			return nil, "", fmt.Errorf("marshal xiaomi tool-call params: %w", err)
		}
		toolCalls = append(toolCalls, xiaomiToolCall{
			name:      funcName,
			arguments: string(argsJSON),
		})
	}

	last := matches[len(matches)-1]
	remaining := content[last[1]:]
	return toolCalls, remaining, nil
}

func isToolWrapperName(name string) bool {
	switch name {
	case "editor", "bash", "agent":
		return true
	default:
		return false
	}
}
