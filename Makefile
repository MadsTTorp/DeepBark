export PYTHONPATH := $(CURDIR)

.PHONY: all run-pipeline clean

# Define the test output directory.
TEST_DIR = test_output

# Default target: run the pipeline.
all: run-pipeline

# Run the entire pipeline with the test output directory.
run-pipeline:
	@echo "Creating test output directory $(TEST_DIR)..."
	mkdir -p $(TEST_DIR)
	@echo "Running the Dog Breed Pipeline locally..."
	python src/pipeline/pipeline_creation.py \
		--scrape-output-path $(TEST_DIR) \
		--scrape-output-file scraped_breeds.parquet \
		--document-output-path $(TEST_DIR) \
		--document-output-file breed_documents.parquet \
		--index-output-path $(TEST_DIR)
	@echo "Pipeline execution complete. Check the $(TEST_DIR) folder for outputs."

# Clean up the generated test outputs.
clean:
	@echo "Cleaning test output directory $(TEST_DIR)..."
	rm -rf $(TEST_DIR)
	@echo "Clean complete."
