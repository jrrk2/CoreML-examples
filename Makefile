# Unified Makefile for CoreML Llama Tools
# Usage:
#   make all           - Build all tools
#   make llamademo     - Build main demo
#   make debug_loader  - Build debug tool
#   make model_compiler - Build model compiler
#   make clean         - Clean all targets

# Compiler settings
CXX = clang++
OBJCXX_FLAGS = -std=c++17 -stdlib=libc++ -fobjc-arc -x objective-c++
FRAMEWORKS = -framework Foundation -framework CoreML -framework Accelerate

# Build configurations
RELEASE_FLAGS = -O2
DEBUG_FLAGS = -O0 -g

# Default target
all: llamademo debug_loader model_compiler debug_inference minimal_test

# Main demo application
llamademo: main.mm LlamaInference.mm LlamaInference.h
	@echo "🔨 Building llamademo (release)..."
	$(CXX) $(OBJCXX_FLAGS) $(RELEASE_FLAGS) $(FRAMEWORKS) -o llamademo main.mm LlamaInference.mm

# Debug inference step-by-step
debug_inference: debug_inference.mm LlamaInference.mm LlamaInference.h
	@echo "🔨 Building debug_inference..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o debug_inference debug_inference.mm LlamaInference.mm

# Minimal CoreML test
minimal_test: minimal_test.mm
	@echo "🔨 Building minimal_test..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o minimal_test minimal_test.mm

# CoreML API method inspector
method_inspector: method_inspector.mm
	@echo "🔨 Building method_inspector..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o method_inspector method_inspector.mm

# Clean stateful model test
stateful_test: stateful_test.mm
	@echo "🔨 Building stateful_test..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o stateful_test stateful_test.mm

# Compile and run in one step
compile_and_run: compile_and_run.mm
	@echo "🔨 Building compile_and_run..."
	$(CXX) $(OBJCXX_FLAGS) $(RELEASE_FLAGS) $(FRAMEWORKS) -o compile_and_run compile_and_run.mm

# Debug loader for troubleshooting
debug_loader: debug_loader.mm
	@echo "🔨 Building debug_loader..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o debug_loader debug_loader.mm

# Model compiler utility
model_compiler: model_compiler.mm
	@echo "🔨 Building model_compiler..."
	$(CXX) $(OBJCXX_FLAGS) $(RELEASE_FLAGS) $(FRAMEWORKS) -o model_compiler model_compiler.mm

# Development/debug build of main demo
llamademo-debug: main.mm LlamaInference.mm LlamaInference.h
	@echo "🔨 Building llamademo (debug)..."
	$(CXX) $(OBJCXX_FLAGS) $(DEBUG_FLAGS) $(FRAMEWORKS) -o llamademo-debug main.mm LlamaInference.mm

# Clean all built targets
clean:
	@echo "🧹 Cleaning all targets..."
	rm -f llamademo debug_loader model_compiler llamademo-debug

# Install/check dependencies
install:
	@echo "📋 Checking dependencies..."
	@echo "Ensure you have Xcode command line tools installed:"
	@echo "  xcode-select --install"
	@command -v clang++ >/dev/null 2>&1 || { echo "❌ clang++ not found"; exit 1; }
	@echo "✅ Dependencies OK"

# Help target
help:
	@echo "CoreML Llama Tools - Available targets:"
	@echo ""
	@echo "  all              Build all tools (default)"
	@echo "  llamademo        Build main inference demo"
	@echo "  debug_loader     Build model loading debugger"
	@echo "  model_compiler   Build model compilation utility"
	@echo "  llamademo-debug  Build debug version of main demo"
	@echo "  clean            Remove all built executables"
	@echo "  install          Check dependencies"
	@echo "  help             Show this help"
	@echo ""
	@echo "Usage examples:"
	@echo "  make model_compiler"
	@echo "  ./model_compiler model.mlpackage"
	@echo "  make llamademo"
	@echo "  ./llamademo compiled_model.mlmodelc \"Hello!\""

# Test targets (if models are available)
test-debug: debug_loader
	@echo "🧪 Testing debug loader..."
	@if [ -f "StatefulMistral7BInstructFP16.mlpackage" ]; then \
		./debug_loader StatefulMistral7BInstructFP16.mlpackage; \
	else \
		echo "⚠️  No test model found"; \
	fi

test-compile: model_compiler
	@echo "🧪 Testing model compiler..."
	@if [ -f "StatefulMistral7BInstructFP16.mlpackage" ]; then \
		./model_compiler StatefulMistral7BInstructFP16.mlpackage; \
	else \
		echo "⚠️  No test model found"; \
	fi

test-run: compile_and_run
	./$< llama-2-7b-chat.mlpackage "Hello, how are you?"

# Add to your Makefile
interactive_engine: interactive_llama_engine.mm
	@echo "🔨 Building interactive_llama_engine..."
	$(CXX) $(OBJCXX_FLAGS) $(RELEASE_FLAGS) $(FRAMEWORKS) -o $@ $<

test-inter: interactive_engine
	./$< llama-2-7b-chat.mlpackage

sliding_window_llama: sliding_window_llama.mm
	@echo "🔨 Building $@..."
	$(CXX) $(OBJCXX_FLAGS) $(RELEASE_FLAGS) $(FRAMEWORKS) -o $@ $<

# Then run
test-slide: sliding_window_llama
	./$< llama-2-7b-chat.mlpackage

# Mark phony targets
.PHONY: all clean install help test-debug test-compile
