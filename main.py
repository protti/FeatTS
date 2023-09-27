from src import FeatTS

if __name__ == '__main__':
    listDataset = ['Coffee']
    for nameDataset in listDataset:
        print(FeatTS.FeatTS(nameDataset,True))