# /************************************************************************************
# ***
# ***	File Author: Dell, 2018-11-12 20:14:49
# ***
# ************************************************************************************/


all:
	python setup.py install --record install.log

clean:
	rm -rf build dist __pycache__
	rm -rf *egg-info install.log

example:
	cd test && python test.py girl.jpg && cd ..

uninstall:
	cat install.log | xargs rm -rf
