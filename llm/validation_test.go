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
		name             string
		cacheType        string
		isEmbeddingModel bool
		want             string
		wantErr          bool
	}{
		{
			name:             "valid type for normal model with q4_0 kv",
			cacheType:        "q4_0",
			isEmbeddingModel: false,
			want:             "q4_0",
			wantErr:          false,
		},
		{
			name:             "valid type for normal model with q8_0 kv",
			cacheType:        "q8_0",
			isEmbeddingModel: false,
			want:             "q8_0",
			wantErr:          false,
		},
		{
			name:             "invalid type",
			cacheType:        "invalid",
			isEmbeddingModel: false,
			want:             "f16",
			wantErr:          false,
		},
		{
			name:             "quantized type for embedding model",
			cacheType:        "q8_0",
			isEmbeddingModel: true,
			want:             "f16",
			wantErr:          false,
		},
		{
			name:             "valid type for embedding model",
			cacheType:        "f16",
			isEmbeddingModel: true,
			want:             "f16",
			wantErr:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ValidateKVCacheType(tt.cacheType, tt.isEmbeddingModel)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
			}
			if tt.want != got {
				t.Errorf("expected %v, got %v", tt.want, got)
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
