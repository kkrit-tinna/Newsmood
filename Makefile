.PHONY: demo

demo:
	pip install -e . && \
	tmp=$$(mktemp -d) && \
	NEWSMOOD_STORE__DB_PATH=$$tmp/newsmood.db \
	NEWSMOOD_REPORT__OUTPUT_DIR=$$tmp/reports \
	newsmood report --offline && \
	echo "report in $$tmp/reports"
