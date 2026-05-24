package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResponsesReasoningMetadata_JSONRoundtrip(t *testing.T) {
	t.Parallel()

	original := ResponsesReasoningMetadata{ItemID: "resp-item-99"}
	data, err := json.Marshal(original)
	require.NoError(t, err)

	var restored ResponsesReasoningMetadata
	require.NoError(t, json.Unmarshal(data, &restored))
	require.Equal(t, original.ItemID, restored.ItemID)
}

func TestResponsesReasoningMetadata_JSONRoundtrip_plainObject(t *testing.T) {
	t.Parallel()

	plain := []byte(`{"item_id":"resp-plain"}`)
	var restored ResponsesReasoningMetadata
	require.NoError(t, json.Unmarshal(plain, &restored))
	require.Equal(t, "resp-plain", restored.ItemID)
}
