import random
import copy
random.seed(42)
import csv
import torch
import time
import statistics
import wandb


def byGuide(data, val=None, test=None):
    val_guides = val
    if val == None:
        val_guides = [
        "GGGTGGGGGGAGTTTGCTCCTGG",
        "GACCCCCTCCACCCCGCCTCCGG",
        "GGCCTCCCCAAAGCCTGGCCAGG",
        "GAACACAAAGCATAGACTGCGGG"
        
        ]
    test_guides = test
    if test==None:
        test_guides = [
            "GCAAAACTCAACCCTACCCCAGG",
            "GGCCCAGACTGAGCACGTGATGG",
            "GGGAAAGACCCAGCATCCGTGGG",
            "GGAATCCCTTCTGCAGCACCTGG",
            "GTGAGTGAGTGTGTGCGTGTGGG",
            "GATGATGATGCCCCGGGCGTTGG",
            "GCCGGAGGGGTTTGCACAGAAGG"
        ]
    
    train_set =  []
    val_set = []
    test_set = []
    for pair in data:
        pair['off'] = torch.tensor([1., 0.])
        if pair['grna_target_sequence'] in val_guides:
            val_set.append(pair)
        elif pair['grna_target_sequence'] in test_guides:
            test_set.append(pair)
        else:   
            train_set.append(pair)
    return [train_set, val_set, test_set]        

def byTarget(data, train=.7, val=.1, test=.2):
    random.shuffle(data)
    train_set =  []
    val_set = []
    test_set = []
    for i in range(len(data)):
        if i <= len(data) * train:
            train_set.append(data[i])
        elif i <= len(data) * (train + val):
            val_set.append(data[i])
        else:
            test_set.append(data[i])
    return [train_set, val_set, test_set]           




def byStudy(data, val=None, test=None):
    val_studies = val
    if val == None:
        val_studies = [
            
            
        ]
    test_studies = test
    if test==None:
        test_studies = [
            
        ]
    train_set =  []
    val_set = []
    test_set = []
    for pair in data:
        pair['off'] = torch.tensor([1., 0.])
        if pair['study_name'] in val_studies:
            val_set.append(pair)
        elif pair['study_name'] in test_studies:
            test_set.append(pair)
        else:   
            train_set.append(pair)
    return [train_set, val_set, test_set]  



def one_hot(data, sign='+'):
    sins = None
    sequence = None
    data = data.lower()
    for n in data:
        
        one_hot = torch.zeros((1, 4))
        if n =='a':
            one_hot[0][0] = 1
        elif n == 'c':
            one_hot[0][1] = 1
        elif n == 'g':
            one_hot[0][2] = 1
        elif n == 't':
            one_hot[0][3] = 1
        if sins == None:
            sequence = copy.deepcopy(one_hot)
            sins = 1
        else:
            sequence = torch.cat((sequence, one_hot), dim=0)
    if list(sequence.size())[0] < 23:
        for i in range(23 - list(sequence.size())[0]):
            sequence = torch.cat((sequence, torch.zeros((1, 4))), dim=0)  
    if list(sequence.size())[0] > 23: 
        sequence = sequence[:23]
    if sign == '-':
        sequence = torch.flip(sequence, [1])      
    return sequence   

        


def dataLoader(file="crisprsql.csv", batch=64, mode="target"):
    ftime = time.monotonic()
    with open(file) as f:
        d = list(csv.DictReader(f))
        if mode == "study":
            loadData = byStudy(d)
        elif mode == "guide":
            loadData = byGuide(d)
        else:
            loadData = byTarget(d)
    data = list()
    
    for t in range(3):
        average_value = list()
        thisdata = list()
        for line in loadData[t]:
            if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
                thisdata.append([
                    [one_hot(line['grna_target_sequence'], line['grna_target_strand']).unsqueeze_(0).unsqueeze_(0), 
                        one_hot(line['target_sequence'], line["target_strand"]).unsqueeze_(0).unsqueeze_(0)],
                    torch.tensor([[float(line['cleavage_freq']) + 0.0000000001]])])
                average_value.append(float(line['cleavage_freq']))   
        # mode = 0
        # zero = 0
        # for p in average_value:
        #     if p == statistics.mode(average_value):
        #         mode+=1
        #     if p <0:
        #         zero+=1 
        # print(f"average CFD of {len(average_value)} datapoints in set {t + 1}: {sum(average_value)/len(average_value)}.\nMedian: {statistics.median(average_value)}.\nMode: {statistics.mode(average_value)} with {mode} datapoint.\nstandard deviation: {statistics.pstdev(average_value)}.\nlowest value: {min(average_value)}.\nHighest value: {max(average_value)}\n{zero} datapoints below zero\n\n")
        thisdata1 = list()            
        for i in range(int(len(thisdata)/batch)):
            ones = None
            twos = None
            threes = None
            for j in range(batch):
                
                if  ones == None:
                    ones = thisdata[(i * batch) + j][0][0]
                    twos = thisdata[(i * batch) + j][0][1]
                    threes = thisdata[(i * batch) + j][1]
                else:
                    ones = torch.cat((ones, thisdata[(i * batch) + j][0][0]), dim=0)  
                    twos = torch.cat((twos, thisdata[(i * batch) + j][0][1]), dim=0) 
                    threes = torch.cat((threes, thisdata[(i * batch) + j][1]), dim=0)      
                       
            thisdata1.append([[ones, twos], threes])          


        data.append(thisdata1) 
 
    print('time to load data: ', time.monotonic() - ftime, 'seconds')   
    return data
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
def roc(labels, outputs):
    llabels = labels.tolist()
    loutputs = outputs.tolist()
    average_values = dict()
    # print(len(llabels), len(loutputs))
    for i in range(30, 31):
        thislabel = list()
        thisoutput = list()
        pres = 0
        totalpres = 0
        for j in range(len(llabels)):

            if llabels[j][0] <= .01 / i:
                thislabel.append(0)
            else:
                thislabel.append(1)    
            if loutputs[j][0] <= .01 / i:
                thisoutput.append(0)
            else:
                thisoutput.append(1)
            if thislabel[-1] == thisoutput[-1]:
                pres += 1
            totalpres +=1        
        lr_precision, lr_recall, _ = precision_recall_curve(thislabel, thisoutput)
        average_values[.1/i] = [roc_auc_score(thislabel, thisoutput), auc(lr_recall, lr_precision), pres/totalpres]
    return average_values    


                


                
            


def Train(epochs, optim, crit, batch_per, train_data, val_data, net, device, optim_time=None, logpath=None):
    net.to(device)
    #def optim, loss, and init graph data
    criterion = crit
    optimizer = optim
    full_full_labels = None
    for i, data in enumerate(train_data, 0):
        if full_full_labels == None:
            full_full_labels = data[1].to(device) 
        else:
            full_full_labels = torch.cat((full_full_labels, data[1].to(device)), 0)   
    full_val_labels = None         
    for i, data in enumerate(val_data, 0):
        if full_val_labels == None:
            full_val_labels = data[1].to(device) 
        else:
            full_val_labels = torch.cat((full_val_labels, data[1].to(device)), 0)            
    print("begin training")
    x = []
    y = []
    valx = []
    valy = []
    corx = []
    corvalx = []
    cory = []
    corvaly = []
    if logpath!= None:
        f = open(logpath, 'w')
    #these go down, and random loss is ~2.303 so 15 will be replaced
    best = 15
    bestval = 15
    e = 0
    times = list()
    for epoch in range(epochs):  # loop over the dataset multiple times
        for q in optim_time:
            if epoch+1 == q:
                optimizer = optim_time[q]
                # net = copy.deepcopy(bestnet)
        ftime = time.monotonic()
        random.shuffle(train_data)
        correct = 0
        total = 0
        running_loss = 0.0
        # train mode
        net.train()
        full_output = None
        full_labels = None
        full_full_output = None
        for i, data in enumerate(train_data, 0):
            inputs, labels = data[0], data[1].to(device) 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)   
            correct += (predicted == labels).sum().item()
            # print statistics
            running_loss += loss.item()
            if full_output == None:
                full_output = outputs
            else:
                full_output = torch.cat((full_output, outputs), 0)

            if full_labels == None:
                full_labels = labels
            else:
                full_labels = torch.cat((full_labels, labels), 0)     


            if i % batch_per == batch_per - 1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (e + 1, i + 1, running_loss / batch_per))
                    # best = min(best, running_loss / batch_per)
                    
                    # print('Accuracy of the network on the ' + str(batch_per) + 'th update: %d %%' % (
                    #     100 * correct / total))
                    
                    wl = roc(full_labels, full_output)
                    wandlog = {}
                    for q in wl:
                        wandlog[f"{q} ROC_AUC"] = wl[q][0]
                        wandlog[f"{q} PR_AUC"] = wl[q][1]
                        wandlog[f"{q} ACCURACY"] = wl[q][2]



                    wandlog.update({
                        "LOSS": running_loss / batch_per, 
                        "TYPE": "TRAIN", 
                        'EPOCH': e+1, 
                        'UPDATE': (e*len(train_data)) + i + 1})
                    wandb.log(wandlog)
                    if full_full_output == None:
                        full_full_output = full_output
                        # print('hello')
                    else:
                        # print("yo")
                        full_full_output = torch.cat((full_full_output, full_output), 0)  
                    
                    full_output = None
                    full_labels = None


                    running_loss = 0
                    correct = 0
                    total = 0
    
        if total != 0:
            print('[%d] loss: %.20f' %
            (epoch + 1, running_loss / total))

        # if logpath != None:
        #     f.write('[%d] loss: %.20f' %
        # (epoch + 1, running_loss / total))   
        # x.append((epoch * len(train_data)) + i)
        # y.append(running_loss/batch_per)
        if full_full_output == None:
            full_full_output = full_output
        else:
            full_full_output = torch.cat((full_full_output, full_output), 0)  
        wl = roc(full_full_labels, full_full_output)
        wandlog = {}
        for q in wl:
            wandlog[f"{q} ROC_AUC"] = wl[q][0]
            wandlog[f"{q} PR_AUC"] = wl[q][1]
            wandlog[f"{q} ACCURACY"] = wl[q][2]
        wandlog.update({
            "LOSS": running_loss / batch_per, 
            "TYPE": "TRAIN", 
            'EPOCH': e+1, 
            'UPDATE': (e + 1) *len(train_data)})           
        wandb.log(wandlog) 

        full_output = None
        full_full_output = None
        running_loss = 0
        # cory.append(100 * correct / total)   
        # corx.append((epoch * len(train_data)) + i) 
        correct = 0
        total = 0           
        running_loss = 0
        net.eval()
        correct = 0
        total = 0
        #check val set
        for i, data in enumerate(val_data, 0):
            inputs, labels = data[0], data[1].to(device) 
            outputs = net(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            running_loss += loss.item()
            total+= labels.size(0)   
            if full_output == None:
                full_output = outputs
            else:
                full_output = torch.cat((full_output, outputs), 0)  
        print(f'Validation loss for Epoch [{e +1}]: {running_loss/total}') 
        # if logpath != None:
        #     f.write(f'Validation loss for Epoch [{epoch}]: {running_loss/total}')  
            
        wl = roc(full_val_labels, full_output)
        wandlog = {}
        for q in wl:
            wandlog[f"{q} ROC_AUC"] = wl[q][0]
            wandlog[f"{q} PR_AUC"] = wl[q][1]
            wandlog[f"{q} ACCURACY"] = wl[q][2]
        wandlog.update({
            "LOSS": running_loss / len(val_data), 
            "TYPE": "VAL", 
            'EPOCH': e+1, 
            'UPDATE': (e + 1)*len(train_data)})           
        wandb.log(wandlog) 
        # best = min(best, running_loss / total)
        # bestval = min(bestval, running_loss / total)
        if ((running_loss/total) - bestval) >= 0:
            curr = 0
            lastcurr = 0
            oldepoch = epoch
            for q in optim_time:
                curr+=q
                if epoch > lastcurr and epoch < curr:
                    epoch = curr
                lastcurr += q    
            print('Early Stop')
            print(f"Best Validation loss: {bestval}")
            print(f"Current Validation loss: {running_loss / total}")
            
            return
        else:
            bestval = running_loss / total

        running_loss = 0
        correct = 0
        total = 0
        times.append(time.monotonic() - ftime)
        PATH = f'.net.pth'
        torch.save(net.state_dict(), PATH)
        print('time for epoch: ', times[-1], 'seconds')
        if logpath != None:
            f.write(f'time for epoch: {times[-1]}, seconds') 
        e+=1
    


            





    # finish training. in future dont plot and save here just return them
    print('Finished Training')
    print('average time per epoch: ', sum(times)/len(times), 'seconds')
    if logpath != None:
            f.write('Finished Training')
            f.write(f'average time per epoch: {sum(times)/len(times)} seconds')
            f.close()
    # ploss = plt.figure()
    # ploss.plot(x, y, label = "train")
    # ploss.plot(valx, valy, label = "valid")
    # ploss.legend()
    # ploss.ylabel('Running Loss')
    # ploss.xlabel('Updates')
    # acur = plt.figure()
    # acur.plot(corx, cory, label = "train")
    # acur.plot(corvalx, corvaly, label = "valid")
    # acur.legend()
    # acur.ylabel('Accuracy')
    # acur.xlabel('Updates')
    
    return 


def Test(net, dataset, device, crit, logpath=None):
    
    net.eval()
    correct = 0
    total = 0
    totalloss = 0
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):
            inputs, labels = data[0], data[1].to(device) 
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            totalloss+=1
            correct += (predicted == labels).sum().item()
            loss+=crit(outputs, labels)
    if logpath!= None:
        f = open(logpath, 'w')
        f.write('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        f.write(f"total: {total} correct: {correct}")
        f.write(f'loss: {loss/totalloss}')
        f.close()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print(f"total: {total} correct: {correct}") 
    print(f'loss: {loss/totalloss}')
    return 100 * correct / total        

def getAllStudy():
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()
        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"]:
                try:
                    alls[row['study_name']].add(row['grna_target_sequence'])   
                except KeyError:
                    alls[row["study_name"]] = set(row['grna_target_sequence'])    
        for r in alls:
            print(r)
            print(alls[r])
            print(len(alls[r]))
        

def getallGuide():
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()

        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"]:
                try:
                    alls[row['grna_target_sequence']].add(row['target_sequence'])   
                except KeyError:
                    alls[row["grna_target_sequence"]] = set(row['target_sequence'])    
        for r in alls:
            print(r)
            print(alls[r])
            print(len(alls[r]))
        

def aboveandbelow(threshold):
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()
        above = 0
        total = 0
        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"] and row['cleavage_freq'] != '':
                if float(row['cleavage_freq']) > threshold:
                    above+=1
                total+=1
    

    print(f'Above: {above / total}%. Below: {(total - above) / total}')







def NewTrain(epochs, optim, crit, batch_per, train_data, val_data, net, device, optim_time=None, logpath=None):
    net.to(device)
    #def optim, loss, and init graph data
    criterion = crit
    optimizer = optim
    # get all labels for ROC
    full_full_labels = None
    for i, data in enumerate(train_data, 0):
        if full_full_labels == None:
            full_full_labels = data[1].to(device) 
        else:
            full_full_labels = torch.cat((full_full_labels, data[1].to(device)), 0)   
    full_val_labels = None         
    for i, data in enumerate(val_data, 0):
        if full_val_labels == None:
            full_val_labels = data[1].to(device) 
        else:
            full_val_labels = torch.cat((full_val_labels, data[1].to(device)), 0)            
    print("begin training")
    if logpath!= None:
        f = open(logpath, 'w')
    #these go down, and random loss is ~2.303 so 15 will be replaced
    best = 15
    bestval = 15
    e = 0
    # begin training loop, larget loop is for lr scedule
    times = list()
    for q in optim_time:
        optimizer = optim_time[q]
        print(q)
        # net = copy.deepcopy(bestnet)
        # epoch loop
        for epoch in range(q):  # loop over the dataset multiple times
            ftime = time.monotonic()
            random.shuffle(train_data)
            correct = 0
            total = 0
            running_loss = 0.0
            # train mode
            net.train()
            full_output = None
            full_labels = None
            full_full_output = None
            
            for i, data in enumerate(train_data, 0):
                # train step
                inputs, labels = data[0], data[1].to(device) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)   
                correct += (predicted == labels).sum().item()
                
                running_loss += loss.item()
                if full_output == None:
                    full_output = outputs
                else:
                    full_output = torch.cat((full_output, outputs), 0)

                if full_labels == None:
                    full_labels = labels
                else:
                    full_labels = torch.cat((full_labels, labels), 0)     

                # print statistics
                if i % batch_per == batch_per - 1:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                            (e + 1, i + 1, running_loss / batch_per))
                        # best = min(best, running_loss / batch_per)
                        
                        # print('Accuracy of the network on the ' + str(batch_per) + 'th update: %d %%' % (
                        #     100 * correct / total))
                        
                        wl = roc(full_labels, full_output)
                        wandlog = {}
                        for q in wl:
                            wandlog[f"{q} ROC_AUC"] = wl[q][0]
                            wandlog[f"{q} PR_AUC"] = wl[q][1]
                            wandlog[f"{q} ACCURACY"] = wl[q][2]



                        wandlog.update({
                            "LOSS": running_loss / batch_per, 
                            "TYPE": "TRAIN", 
                            'EPOCH': e+1, 
                            'UPDATE': (e*len(train_data)) + i + 1})
                        wandb.log(wandlog)
                        if full_full_output == None:
                            full_full_output = full_output
                        else:
                            full_full_output = torch.cat((full_full_output, full_output), 0)  
                        
                        full_output = None
                        full_labels = None


                        running_loss = 0
                        correct = 0
                        total = 0
            # print('[%d] loss: %.20f' %
            # (epoch + 1, running_loss / total))
            # if logpath != None:
            #     f.write('[%d] loss: %.20f' %
            # (epoch + 1, running_loss / total))   
            if full_full_output == None:
                full_full_output = full_output
            else:
                full_full_output = torch.cat((full_full_output, full_output), 0)  
            # ROC is commented out when training on 10 samples
            # wl = roc(full_full_labels, full_full_output)
            wandlog = {}
            # for q in wl:
            #     wandlog[f"{q} ROC_AUC"] = wl[q][0]
            #     wandlog[f"{q} PR_AUC"] = wl[q][1]
            #     wandlog[f"{q} ACCURACY"] = wl[q][2]
            wandlog.update({
                "LOSS": running_loss / batch_per, 
                "TYPE": "TRAIN", 
                'EPOCH': e+1, 
                'UPDATE': (e + 1) *len(train_data)})           
            wandb.log(wandlog) 

            full_output = None
            full_full_output = None
            running_loss = 0
            correct = 0
            total = 0           
            running_loss = 0
            net.eval()
            correct = 0
            total = 0
            #check val set
            for i, data in enumerate(val_data, 0):
                inputs, labels = data[0], data[1].to(device) 
                outputs = net(inputs)
                loss = criterion(outputs, labels) 
                loss.backward()
                running_loss += loss.item()
                total+= labels.size(0)   
                if full_output == None:
                    full_output = outputs
                else:
                    full_output = torch.cat((full_output, outputs), 0)  
            print(f'Validation loss for Epoch [{e +1}]: {running_loss/total}') 
            # if logpath != None:
            #     f.write(f'Validation loss for Epoch [{epoch}]: {running_loss/total}')  
            
            # wl = roc(full_val_labels, full_output)
            wandlog = {}
            # for q in wl:
            #     wandlog[f"{q} ROC_AUC"] = wl[q][0]
            #     wandlog[f"{q} PR_AUC"] = wl[q][1]
            #     wandlog[f"{q} ACCURACY"] = wl[q][2]
            wandlog.update({
                "LOSS": running_loss / len(val_data), 
                "TYPE": "VAL", 
                'EPOCH': e+1, 
                'UPDATE': (e + 1)*len(train_data)})           
            wandb.log(wandlog) 
            # best = min(best, running_loss / total)
            # early stop just goes to the next lr change checkpoint
            bestval = min(bestval, running_loss / total)
            if bestval != running_loss / total or bestval < running_loss / total:
                        print('Early Stop')
                        print(f"Best Validation loss: {bestval}")
                        print(f"Current Validation loss: {running_loss / total}")
                        
                        e+=1
                        break

            running_loss = 0
            correct = 0
            total = 0
            times.append(time.monotonic() - ftime)
            PATH = f'.net.pth'
            torch.save(net.state_dict(), PATH)
            print('time for epoch: ', times[-1], 'seconds')
            if logpath != None:
                f.write(f'time for epoch: {times[-1]}, seconds') 
            e+=1
        


            





    # finish training. in future dont plot and save here just return them
    print('Finished Training')
    print('average time per epoch: ', sum(times)/len(times), 'seconds')
    if logpath != None:
            f.write('Finished Training')
            f.write(f'average time per epoch: {sum(times)/len(times)} seconds')
            f.close()
    
    return 
