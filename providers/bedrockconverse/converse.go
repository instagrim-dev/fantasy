// Package bedrockconverse provides an implementation of the fantasy AI SDK
// for AWS Bedrock models using the Converse API. Unlike the bedrock provider
// (which wraps the Anthropic Messages API), this provider uses the
// model-agnostic Converse/ConverseStream endpoints and supports all Bedrock
// foundation models (Nova, Llama, Mistral, Cohere, etc.).
package bedrockconverse

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const (
	// Name is the name of the Bedrock Converse provider.
	Name = "bedrockconverse"
)

// Option defines a function that configures provider options.
type Option func(*options)

// HTTPClient is the interface for an HTTP client used by the provider.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

type options struct {
	region    string
	awsConfig *aws.Config
	name      string
}

type provider struct {
	options options
}

// New creates a new Bedrock Converse provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	o.name = cmp.Or(o.name, Name)

	return &provider{options: o}, nil
}

// WithRegion sets the AWS region for the provider.
func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
	}
}

// WithAWSConfig injects a pre-built AWS config, skipping LoadDefaultConfig.
func WithAWSConfig(cfg aws.Config) Option {
	return func(o *options) {
		o.awsConfig = &cfg
	}
}

// WithName overrides the provider name (default: "bedrockconverse").
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// Name implements fantasy.Provider.
func (p *provider) Name() string {
	return p.options.name
}

// LanguageModel implements fantasy.Provider.
func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	cfg, err := p.resolveConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("bedrockconverse: resolve AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(cfg)
	return &languageModel{
		client:       client,
		modelID:      modelID,
		providerName: p.options.name,
	}, nil
}

func (p *provider) resolveConfig(ctx context.Context) (aws.Config, error) {
	if p.options.awsConfig != nil {
		cfg := *p.options.awsConfig
		if p.options.region != "" {
			cfg.Region = p.options.region
		}
		return cfg, nil
	}

	var loadOpts []func(*awsconfig.LoadOptions) error
	if p.options.region != "" {
		loadOpts = append(loadOpts, awsconfig.WithRegion(p.options.region))
	}
	return awsconfig.LoadDefaultConfig(ctx, loadOpts...)
}

type languageModel struct {
	client       *bedrockruntime.Client
	modelID      string
	providerName string
}

// Provider implements fantasy.LanguageModel.
func (m *languageModel) Provider() string { return m.providerName }

// Model implements fantasy.LanguageModel.
func (m *languageModel) Model() string { return m.modelID }

// Generate implements fantasy.LanguageModel.
func (m *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	system, messages, err := mapPrompt(call.Prompt)
	if err != nil {
		return nil, fmt.Errorf("bedrockconverse: map prompt: %w", err)
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(m.modelID),
		Messages: messages,
	}
	if len(system) > 0 {
		input.System = system
	}
	if tools := mapTools(call.Tools); len(tools) > 0 {
		input.ToolConfig = &types.ToolConfiguration{
			Tools: tools,
		}
		if tc := mapToolChoice(call.ToolChoice); tc != nil {
			input.ToolConfig.ToolChoice = tc
		}
	}
	if ic := mapInferenceConfig(call); ic != nil {
		input.InferenceConfig = ic
	}

	out, err := m.client.Converse(ctx, input)
	if err != nil {
		return nil, wrapAWSError(err)
	}

	return mapResponse(out), nil
}

// Stream implements fantasy.LanguageModel.
func (m *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	system, messages, err := mapPrompt(call.Prompt)
	if err != nil {
		return nil, fmt.Errorf("bedrockconverse: map prompt: %w", err)
	}

	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(m.modelID),
		Messages: messages,
	}
	if len(system) > 0 {
		input.System = system
	}
	if tools := mapTools(call.Tools); len(tools) > 0 {
		input.ToolConfig = &types.ToolConfiguration{
			Tools: tools,
		}
		if tc := mapToolChoice(call.ToolChoice); tc != nil {
			input.ToolConfig.ToolChoice = tc
		}
	}
	if ic := mapInferenceConfig(call); ic != nil {
		input.InferenceConfig = ic
	}

	out, err := m.client.ConverseStream(ctx, input)
	if err != nil {
		return nil, wrapAWSError(err)
	}

	return newStreamResponse(out), nil
}

// GenerateObject implements fantasy.LanguageModel.
func (m *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return object.GenerateWithTool(ctx, m, call)
}

// StreamObject implements fantasy.LanguageModel.
func (m *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return object.StreamWithTool(ctx, m, call)
}

// wrapAWSError converts AWS SDK errors into fantasy.ProviderError.
func wrapAWSError(err error) error {
	if err == nil {
		return nil
	}

	var ae interface {
		error
		ErrorCode() string
		ErrorMessage() string
	}
	if errors.As(err, &ae) {
		statusCode := 500
		switch ae.ErrorCode() {
		case "ThrottlingException", "TooManyRequestsException":
			statusCode = 429
		case "ValidationException":
			statusCode = 400
		case "AccessDeniedException":
			statusCode = 403
		case "ResourceNotFoundException", "ModelNotReadyException":
			statusCode = 404
		case "ServiceUnavailableException", "InternalServerException":
			statusCode = 500
		case "ModelStreamErrorException", "ModelErrorException":
			statusCode = 502
		case "ModelTimeoutException":
			statusCode = 504
		}
		return &fantasy.ProviderError{
			Message:    ae.ErrorMessage(),
			Title:      fantasy.ErrorTitleForStatusCode(statusCode),
			StatusCode: statusCode,
			Cause:      err,
		}
	}

	return &fantasy.ProviderError{
		Message:    err.Error(),
		Title:      "bedrock converse error",
		StatusCode: 500,
		Cause:      err,
	}
}
