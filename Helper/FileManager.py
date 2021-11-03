import csv

class FileManager():

    def __init__(self):
        self.file_csv = None

    def create_file(self, name):
        self.file_csv = open(name + '.csv', 'w', 1)
        self.writer  = csv.writer(self.file_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(['P', 'Q', 'Param01', 'Param02', 'Avg Metric', 'Time'])
        self.file_csv.flush()

    def write2file(self, param, p, q, metric, time):
        self.writer.writerow([p, q, param[0], param[1], metric, time])
        self.file_csv.flush()

    def close_file(self):
        self.file_csv.close()
