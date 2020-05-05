import json
import os

# Loop through files with news articles in them, read them in.  If they meet criteria, save to other directory
raw_folder_path = '/Users/petermehler/news-please-repo/data/2020/05/04'
renamed_folder_path = '/Users/petermehler/Desktop/CompSci/CS311/Final/news_files/'

# Thanks StackOverFlow user ChrisProsser for the iteration code
for subdir, dirs, files in os.walk(raw_folder_path):
	for file in files:
		if(file != '.DS_Store'):
			news_agency = os.path.basename(subdir)
			with open(os.path.join(subdir, file), 'r') as f:
				d = json.load(f)
				if (d['date_publish'] is not None and d['maintext'] is not None):
					date = d['date_publish'][0:10]
					title = d['title']
					date = date.replace(" ", "")
					title = title.replace(" ", "")
					title = title.replace("/", "")
					print('PATH: ' + os.path.join(subdir, file))
					os.rename(os.path.join(subdir, file), renamed_folder_path + date + '_' + news_agency + '_' + title + '.json')

print("done")
'''
filter for coronavirus news:
for subdir, dirs, files in os.walk(raw_folder_path):
	for file in files:
		if(file != '.DS_Store'):
			news_agency = os.path.basename(subdir)
			with open(os.path.join(subdir, file), 'r') as f:
				d = json.load(f)
				if (d['date_publish'] is not None and d['maintext'] is not None):
					matches = ["corona", "covid", "virus"]
					text = d['maintext'].lower()
					if any(x in text for x in matches):
						date = d['date_publish'][0:10]
						title = d['title']
						date = date.replace(" ", "")
						title = title.replace(" ", "")
						title = title.replace("/", "")
						print('PATH: ' + os.path.join(subdir, file))
						os.rename(os.path.join(subdir, file), renamed_folder_path + date + '_' + news_agency + '_' + title + '.json')
'''
