import csv
import os
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import numpy as np
import sys
import prediction as predict

MH_new = list()
newcountDict = {}
humidityDict = {}

masterDict = {}

def dataLoader(filePath,columns):
    global masterDict
    masterDict.clear()

    with open(filePath, newline='') as csvfile:
        filereader = csv.DictReader(csvfile)
        for row in filereader:
            if row['State'] not in masterDict:
                masterDict[row['State']]={}

            for column in columns:
                masterDict[row['State']].setdefault(column,[]).append(float(row[column]))

#    //// Example to retrieve one particular column /////
#           masterDict[row['State']].setdefault('humidity',[]).append(int(row['Humidity']))

#    //// Itereting the dictionary element by element ////
#    for state, data in masterDict.items():
#        if state == 'AP':
#            print('number of rows = {}'.format(len(data['newcount'])))
#            print ('State = {}'.format(state))
#            print ('Data = {}'.format(data))


# Method to plot columns for states
# @input: states - Symbol for states as per .csv file
# @input: columns - column names as per .csv file
# @output: Plot the data

def displayColumns(states,columns):
    style.use('ggplot')
    index = 0
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for state,data in masterDict.items():
        if state in states:
            rowCount = len(data[columns[0]])
            x = range(0,rowCount,1)
            for column in columns:
                plt.plot(x,data[column],color=colors[index],label=state+" "+column,linewidth=1.5)
                index += 1
                plt.legend()

    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.grid(True,color='w')
    plt.show()

def normalizeData(columnName,data,state=''):
    lsDict = {'ACP':2,'AN':1,'AP':25,'AS':14,'BR':40,'CG':11,'DN':1,'DL':7,'GA':2,'GJ':26,'HR':10,'HP':4,'JH':14,'KA':28,'KL':20,'MP':29,'MH':48,'MN':2,'MZ':1,'OD':21,'PB':13,'PC':1,'RJ':25,'TG':17,'TR':2,'TN':39,'UK':5,'UTCH':1,'UTJK':5,'UTL':1,'UP':80,'WB':42}

    factor = 1
    if columnName == 'Statewise number of tests':
        if state !='':
            sumVal = sum(lsDict.values())
            factor = lsDict[state]/sumVal
        data = [x * factor for x in data]

    return data

# Will be deprecated soon
def displayNewVSRecoveredVSHumidity(states):
    style.use('ggplot')
    index = 0
    #colors = cm.rainbow(np.linspace(0, 1, 10+len(newcountDict['KL'])))
    colors = cm.rainbow(np.linspace(0, 1, 10))
    index = 0
    for state,data in masterDict.items():
        rowCount = len(data['Statewise increement'])
        x = range(0,rowCount,1)
        if state in states:
            plt.plot(x,data['Statewise increement'],color=colors[index],label=state+' New case',linewidth=1.5)
            index += 1
            plt.legend()

    for state,data in masterDict.items():
        rowCount = len(data['Humidity'])
        x = range(0,rowCount,1)
        if state in states:
            plt.bar(x,data['Humidity'],color=colors[index],label=state+' Humidity',width=0.05)
            index += 1
            plt.legend()

    for state,data in masterDict.items():
        rowCount = len(data['Statewise recovered increment'])
        x = range(0,rowCount,1)
        if state in states:
            plt.bar(x,data['Statewise recovered increment'],color=colors[index],label=state+' Recovered',width=0.5)
            index += 1
            plt.legend()

    for state,data in masterDict.items():
        rowCount = len(data['Temperature'])
        x = range(0,rowCount,1)
        if state in states:
            plt.plot(x,data['Temperature'],color=colors[index],label=state+' Temperature',linewidth=1)
            index += 1
            plt.legend()

    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.grid(True,color='w')
    plt.show()


def main(argv):
    if len(argv) < 1:
        print ('python3 reader.py MP KA ... MN')
        exit()
    os.system('rm ./training_2/*')

    columns = ['New Indian','Humidity','Temperature']

    dataLoader("./data/CData.csv",columns)
    #displayNewVSRecoveredVSHumidity(argv)
    predict.trainAndTest(masterDict,argv)
    dataLoader("./data/PData.csv",columns) 
    predict.futurePrediction(masterDict,argv)    
    displayColumns(argv,columns)

if __name__  == "__main__":
    main(sys.argv[1:])
