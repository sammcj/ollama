package llm

import (
	"testing"

	"github.com/ollama/ollama/discover"
)

// testGGML implements GGMLModel for testing
type testGGML struct {
	kv KV
}

func (g *testGGML) KV() KV {
	return g.kv
}

func TestValidateKVCacheType(t *testing.T) {
	tests := []struct {
		name        string
		cacheType   string
		isEmbedding bool
		expected    string
		expectError bool
	}{
		{
			name:        "empty cache type",
			cacheType:   "",
			isEmbedding: false,
			expected:    "",
			expectError: false,
		},
		{
			name:        "invalid cache type",
			cacheType:   "invalid",
			isEmbedding: false,
			expected:    "f16",
			expectError: false,
		},
		{
			name:        "embedding model with q4_0",
			cacheType:   "q4_0",
			isEmbedding: true,
			expected:    "f16",
			expectError: false,
		},
		{
			name:        "embedding model with f32",
			cacheType:   "f32",
			isEmbedding: true,
			expected:    "f32",
			expectError: false,
		},
		{
			name:        "non-embedding model with q4_0",
			cacheType:   "q4_0",
			isEmbedding: false,
			expected:    "q4_0",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ValidateKVCacheType(tt.cacheType, tt.isEmbedding)
			if tt.expectError && err == nil {
				t.Errorf("expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("expected no error, got %v", err)
			}
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestValidateFlashAttentionSupport(t *testing.T) {
	tests := []struct {
		name               string
		kvData             map[string]any
		gpus               discover.GpuInfoList
		flashAttnRequested bool
		want               bool
	}{
		{
			name: "supported model and hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want:               true,
		},
		{
			name: "embedding model",
			kvData: map[string]any{
				"general.architecture":        "bert",
				"bert.attention.key_length":   uint32(32),
				"bert.attention.value_length": uint32(32),
				"bert.pooling_type":           "mean",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want:               false,
		},
		{
			name: "unsupported hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 6},
			},
			flashAttnRequested: true,
			want:               false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ggml := &testGGML{kv: tt.kvData}
			got := ValidateFlashAttentionSupport(ggml, tt.gpus, tt.flashAttnRequested)
			if tt.want != got {
				t.Errorf("expected %v, got %v", tt.want, got)
			}
		})
	}
}
