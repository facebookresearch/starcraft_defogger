all:
	pip install -r requirements.txt
	python setup.py

clean:
	rm -rf build/*
