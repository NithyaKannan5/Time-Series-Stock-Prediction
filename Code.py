#% pip install keras tensorflow
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import rank
from pyspark.sql import Window

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# % matplotlib inline

config = SparkConf().setAppName('Stock Pred Pipeline').setMaster('local[*]')
spc = SparkContext.getOrCreate(conf=config)
sql_context = SQLContext(spc)
dtf = sql_context.read.csv('dbfs:/FileStore/FileStore/data.csv',
                    header=True,
                    inferSchema=True)

dtf = dtf.withColumn("rank", rank().over(Window.partitionBy().orderBy("date")))
train_dtf = dtf.where("rank <= 1625").drop("rank","Open","High","Low","Volume","OpenInt")
test_dtf = dtf.where("rank > 1575").drop("rank","Open","High","Low","Volume","OpenInt")
from pyspark.sql.functions import col, stddev_samp
maxVal_tr = train_dtf.agg({'Close':'max'}).collect()[0][0]
minVal_tr = train_dtf.agg({'Close':'min'}).collect()[0][0]

trainn_dff = train_dtf.withColumn("ScaledClose",
  (col("Close") - minVal_tr)/(maxVal_tr - minVal_tr))
maxVal_te = test_dtf.agg({'Close':'max'}).collect()[0][0]
minVal_te = test_dtf.agg({'Close':'min'}).collect()[0][0]
testt_dff = test_dtf.withColumn("ScaledClose",
  (col("Close") - minVal_tr)/(maxVal_tr - minVal_tr))
price_col_tr = trainn_dff.select("ScaledClose").rdd.map(lambda x: x[0]).collect()
price_col_te = testt_dff.select("ScaledClose").rdd.map(lambda x: x[0]).collect()
import numpy as np
price_col_arr_tr = np.array(price_col_tr)
price_col_arr_te = np.array(price_col_te)
price_col_arr_tr = price_col_arr_tr.reshape(-1,1)
price_col_arr_te = price_col_arr_te.reshape(-1,1)

def genDataPatt(scaledtraindata,his):   
    x_his_ = []  
    y_price_cur_ = []
    train_leng = scaledtraindata.shape[0]
    dayy = his
    while dayy<train_leng:
        y_cur_price = scaledtraindata[dayy,0]
        cur_range = scaledtraindata[dayy-his:dayy,0]
        x_his_.append(cur_range)
        y_price_cur_.append(y_cur_price)
        dayy+=1
    
    x_his_ = np.array(x_his_)
    y_price_cur_ = np.array(y_price_cur_)
    
    x_his_ = x_his_.reshape(x_his_.shape[0],x_his_.shape[1],1)
    return x_his_,y_price_cur_

patternTrainData = genDataPatt(price_col_arr_tr,50)
tr_X = patternTrainData[0]
tr_Y = patternTrainData[1]
tr_X.shape
tr_Y.shape
patternTestData = genDataPatt(price_col_arr_te,50)
te_X = patternTestData[0]
te_Y = patternTestData[1]
te_X.shape
te_Y = te_Y.reshape(-1,1)
te_Y.shape

from tensorflow.keras.layers import SimpleRNN, LSTM
class RNNStockModel():
 
    loss_function ='mean_squared_error'
    batch_size=32
    num_neu = 50
    model = Sequential()
    def __init__(self,trainX,trainY,epoch):
        self.trainX = trainX
        self.trainY = trainY
        self.epoch = epoch
    
    def buildModel(self):
        RNNStockModel.model = Sequential()
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True,
                                            input_shape = (self.trainX.shape[1],1)))
        
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True))
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = True))
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(SimpleRNN(RNNStockModel.num_neu,
                                            activation='tanh',
                                            return_sequences = False))
        
        RNNStockModel.model.add(Dropout(0.2))
        
        RNNStockModel.model.add(Dense(units=RNNStockModel.num_neu,
                                        activation='tanh'))
        
        RNNStockModel.model.add(Dense(units=1))
        return RNNStockModel.model.summary()
    
    def model_fit(self):
        prev = RNNStockModel.model.fit(self.trainX,self.trainY,
                                        epochs=self.epoch,batch_size=RNNStockModel.batch_size,validation_split=0.2,
                                       )
        return prev
    
    def NeuronsUpdate(self,neu):
        RNNStockModel.num_neu = neu
 
    def EpochUpdate(self,epoch):
        self.epoch = epoch
    
    def BatchSizeUpdate(self,cur_batch_size):
        RNNStockModel.batch_size = cur_batch_size
    
    def model_compile(self):
        RNNStockModel.model.compile(optimizer = Adam(),
                                    loss = RNNStockModel.loss_function)
        return RNNStockModel.model.summary()
    
    def evaluateModel(self, x=None, y=None):
        if x == None:
            x = self.trainX
        if y == None:
            y = self.trainY
        scores = RNNStockModel.model.evaluate(x, y, verbose = 0)
        return scores
class LSTMModel(RNNStockModel):
    RNNStockModel.model = Sequential()
    def __init__(self,trainX,trainY,epoch):
        super().__init__(trainX,trainY,epoch)
    
    def buildModel(self,dense=1):
        RNNStockModel.model = Sequential()
        RNNStockModel.model.add(LSTM(
                                 RNNStockModel.num_neu,
                                 input_shape=(None,1)))
        
        RNNStockModel.model.add(Dense(units=1))
        return RNNStockModel.model.summary()


def plt_gr(og_val,pred_val):
    plt.figure(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.plot(og_val,color="Red",label="Original Stock Price")
    plt.plot(pred_val,color="Blue",label="Predicted Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

for epoch in [16]:
        for btch_sz in [70]:
            for num_neurons in [80]:
                F_Mod = LSTMModel(tr_X,tr_Y,epoch=epoch)
                F_Mod.NeuronsUpdate(num_neurons)
                F_Mod.BatchSizeUpdate(btch_sz)
                F_Mod.buildModel()
                F_Mod.model_compile()
                prev = F_Mod.model_fit()
                
                plt.plot(prev.history['loss'])
                plt.plot(prev.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.show()

                pred_val = F_Mod.model.predict(te_X)
                pred_val2 = spc.parallelize(pred_val).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()      
                og_val = spc.parallelize(te_Y).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()
                diffe = [(a-b)*(a-b) for a,b in zip(pred_val2, og_val)]
                
                import math
                rmse_v = math.sqrt(sum(diffe)/len(diffe))
                print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,btch_sz))
                print("RMSE : {}".format(rmse_v))
                plt_gr(og_val,pred_val2)
#Epochs
rmse_v=[]
for epoch in [2,4,8,16,32,64]:
    F_Mod = LSTMModel(tr_X,tr_Y,epoch)
    F_Mod.NeuronsUpdate(80)
    F_Mod.BatchSizeUpdate(70)
    F_Mod.buildModel()
    F_Mod.model_compile()
    prev = F_Mod.model_fit()
    pred_val = F_Mod.model.predict(te_X)
    pred_val2 = spc.parallelize(pred_val).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()
    og_val = spc.parallelize(te_Y).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()
    diffe = [(a-b)*(a-b) for a,b in zip(pred_val2, og_val)]
    import math
    rmse_v.append(math.sqrt(sum(diffe)/len(diffe)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,btch_sz))

epochs = [2,4,8,16,32,64]
print(rmse_v)
plt.plot()
plt.plot(epochs,rmse_v)
plt.title('rmse vs epoch')
plt.ylabel('rmse')
plt.xlabel('epochs')
plt.show()

#Batch_Size
rmse_v=[]
for btch_sz in [10,20,30,40,50,60,70,80,90,100]:
    F_Mod = LSTMModel(tr_X,tr_Y,epoch=16)
    F_Mod.NeuronsUpdate(80)
    F_Mod.BatchSizeUpdate(btch_sz)
    F_Mod.buildModel()
    F_Mod.model_compile()
    prev = F_Mod.model_fit() 
    pred_val = F_Mod.model.predict(te_X)
    pred_val2 = spc.parallelize(pred_val).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()           
    og_val = spc.parallelize(te_Y).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()
    diffe = [(a-b)*(a-b) for a,b in zip(pred_val2, og_val)]
    import math
    rmse_v.append(math.sqrt(sum(diffe)/len(diffe)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,btch_sz))
btch_sz = [10,20,30,40,50,60,70,80,90,100]
print(rmse_v)
plt.plot()
plt.plot(btch_sz,rmse_v)
plt.title('rmse vs batch_size')
plt.ylabel('rmse')
plt.xlabel('batch_size')
plt.show()

#Neurons
rmse_v=[]
for num_neurons in [40,80,120,160,200,240,280,320]:
    F_Mod = LSTMModel(tr_X,tr_Y,epoch = 16)
    F_Mod.NeuronsUpdate(num_neurons)
    F_Mod.BatchSizeUpdate(70)
    F_Mod.buildModel()
    F_Mod.model_compile()
    prev = F_Mod.model_fit()
    pred_val = F_Mod.model.predict(te_X)
    pred_val2 = spc.parallelize(pred_val).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()           
    og_val = spc.parallelize(te_Y).map(lambda x: x[0]*(maxVal_tr- minVal_tr)+minVal_tr).collect()
    diffe = [(a-b)*(a-b) for a,b in zip(pred_val2, og_val)]
    import math
    rmse_v.append(math.sqrt(sum(diffe)/len(diffe)))
    print("For epoch {}, neurons {} and batch_size {}".format(epoch,num_neurons,btch_sz))

neur = [40,80,120,160,200,240,280,320]
print(rmse_v)
plt.plot()
plt.plot(neur,rmse_v)
plt.title('rmse vs num_neurons')
plt.ylabel('rmse')
plt.xlabel('num_neurons')
plt.show()