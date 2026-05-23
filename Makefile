CXX = nvcc

TAU?=1.0
MONITOR?=10000
Dt?=0.1
C2?=1.0
C4?=0.0
C6?=0.0
C12?=0.0
MONITORCONF?=100000
TEMP?=0.1

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include -I/content/random123/include
FLAGS = --expt-extended-lambda -lcufft -std=c++17 -arch=sm_75 \
-gencode arch=compute_61,code=sm_61 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 \
-DNOMONITOR #-DRANDOM123 #-DNOLOGMONITOR 
PARAMSEW = -DC2=$(C2) -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DNBINS=100 -DDt=$(Dt) -DMONITORCONF=$(MONITORCONF) -DTEMP=$(TEMP)#-DDOUBLE  
PARAMSKPZ = -DC2=$(C2) -DKPZ=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DNBINS=100 -DDt=$(Dt) -DMONITORCONF=$(MONITORCONF) -DTEMP=$(TEMP) #-DDOUBLE  
PARAMSANH = -DC2=$(C2) -DC4=$(C4) -DC6=$(C6) -DC12=$(C12) -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DDt=$(Dt) -DMONITORCONF=$(MONITORCONF) -DTEMP=$(TEMP) #-DDOUBLE  
PARAMSPUREANH = -DC12=1.0 -DTAU=$(TAU) -DMONITOR=$(MONITOR) -DDt=$(Dt) -DNBINS=100 -DMONITORCONF=$(MONITORCONF) -DTEMP=$(TEMP) #-DDOUBLE  
PARAMSALM = -DC2=$(C2) -DC4=$(C4) -DC12=$(C12) -DTAUINFINITO -DMONITOR=$(MONITOR) -DNBINS=100 -DDt=$(Dt) -DMONITORCONF=$(MONITORCONF) -DTEMP=$(TEMP) #-DDOUBLE  

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 

EW: 
	$(CXX) $(FLAGS) $(PARAMSEW) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

KPZ: 
	$(CXX) $(FLAGS) $(PARAMSKPZ) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

ANH: 
	$(CXX) $(FLAGS) $(PARAMSANH) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

ALM: 
	$(CXX) $(FLAGS) $(PARAMSALM) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 


STRONGANH: 
	$(CXX) $(FLAGS) $(PARAMSPUREANH) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

activeinterface: ew.cu Makefile
	$(CXX) $(FLAGS) $(PARAMSEW) ew.cu -o activeinterface $(LDFLAGS) $(INCLUDES) 

update_git:
	git add *.cu Makefile *.h *.sh README.md ; git commit -m "program update"; git push

clean:
	rm -f activeinterface
