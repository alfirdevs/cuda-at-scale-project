TARGET = cuda_batch_image_processing.exe
SRC = main.cu
NVCC = nvcc
NVCCFLAGS = -O2 -std=c++17

all: build

build:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf output
