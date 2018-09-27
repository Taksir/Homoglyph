## Convolutional Neural Network Based Ensemble Approach for Homoglyph Recognition

1. First download the dataset from [here](https://drive.google.com/open?id=1LF4A6daY0loiCYEsp_TzGH5PpNaSZB3a). This dataset contains training and validation set.
2. Go above your current directory
3. Create a folder named "DataSet"
4. Paste the folder named "Custom" (which you just downloaded) inside it. 
5. run: "pip install -r requirements.txt" in your shell.
6. go back to the main directory
7. run: python scratch1.py
8. run: python scratch2.py 
9. After you have properly trained a good model (>98% accuracy on validation set), download final test dataset from [here](https://drive.google.com/open?id=1Vxbsrc9PeMnLTgFBMZC6rWNykxAm66Hz).
10. Code for testing an image can be found as commented out in line: 325-355 in scratch1.py file.
11. Code for testing an image can be found as commented out in line: 261-390 in scratch2.py file.
12. Modify codes accordingly to evaluate on the final test dataset.
13. For running the xfer_1.py and xfer_2.py files, first split files in the "Custom" folders into "Train" and "Test" folder in same directory. Line: 20-21 in xfer_1.py and line: 26-27 in xfer_2.py should be changed according to the number of files that you have put in the folder.
14. run: python xfer_1.py
15. run: python xfer_2.py
16. Evaluate the obtained model on the previous final test dataset.



