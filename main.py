import csv

with open('data.csv',encoding="utf16", newline = '\n') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        print(row)