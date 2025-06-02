# Compiler and flags
CXX = clang++
OBJCXX_FLAGS = -std=c++17 -stdlib=libc++ -fobjc-arc -x objective-c++
FRAMEWORKS = -framework Foundation -framework CoreML -framework Accelerate
OPTIMIZATION = -O2

# Target and sources
TARGET = llamademo
SOURCES = main.mm LlamaInference.mm

# Build rule
$(TARGET): $(SOURCES)
	$(CXX) $(OBJCXX_FLAGS) $(OPTIMIZATION) $(FRAMEWORKS) -o $(TARGET) $(SOURCES)

# Clean rule
clean:
	rm -f $(TARGET)

# Install dependencies (if needed)
install:
	@echo "Ensure you have Xcode command line tools installed:"
	@echo "xcode-select --install"

.PHONY: clean install
