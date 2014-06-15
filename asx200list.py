from lxml import html
import requests
import sys  
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
from PyQt4.QtWebKit import *  
import random
import os.path

class Render(QWebPage):  
	def __init__(self, url):  
		self.app = QApplication(sys.argv)  
		QWebPage.__init__(self)  
		self.loadFinished.connect(self._loadFinished)  
		self.mainFrame().load(QUrl(url))  
		self.app.exec_()  
										  
	def _loadFinished(self, result):  
		self.frame = self.mainFrame()  
		self.app.quit()  
															  

url = 'http://www.asx200.com'
r = Render(url)  
page_html = str(r.frame.toHtml().toUtf8())
tree = html.fromstring(page_html)
codes = tree.xpath('//*[@id="index-container"]/table/tbody/tr/td[2]/span/text()')

limit = int(sys.argv[1])
selected = []
count = 0
while count < limit:
	s = random.choice(codes)
	fname = "close/{s}.AX.csv".format(s=s)
	if s not in selected and os.path.isfile(fname):
		selected.append(fname)
		count = count + 1
for s in selected:
	print s
