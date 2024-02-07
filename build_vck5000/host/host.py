import time
import argparse
import pyxrt
import numpy as np

H1 = 32
W1 = 32
W2 = 32
# A=8
# B=4
# C=4
# X=1
# Y=2
# Z=2
A = 2
B = 2
C = 4
X = 1
Y = 1
Z = 1
MH = X*A*H1
KH = Y*B*W1
NH = Z*C*W2

def main(args):
    
    d = pyxrt.device(0)
    xbin = pyxrt.xclbin(args.xclbin)
    uuid = d.load_xclbin(xbin)
    kernellist = xbin.get_kernels()
    print([i.get_name() for i in kernellist])
    # dma = pyxrt.kernel(d, uuid, "dma")
    dma = pyxrt.kernel(d, uuid, "dma_small")

    TX = int(np.ceil(args.m/MH))
    TY = int(np.ceil(args.k/KH))
    TZ = int(np.ceil(args.n/NH))
    print("TX = {}, TY = {}, TZ = {}".format(TX, TY, TZ))
    M = TX*MH
    K = TY*KH
    N = TZ*NH

    wgt = np.random.rand(K,M).astype(np.float32)
    inp = np.random.rand(N,K).astype(np.float32)
    res = inp @ wgt

    boHandle1 = pyxrt.bo(d, wgt.nbytes, pyxrt.bo.normal, 0)
    boHandle2 = pyxrt.bo(d, inp.nbytes, pyxrt.bo.normal, 0)
    boHandle3 = pyxrt.bo(d, res.nbytes, pyxrt.bo.normal, 0)
    bo1 = np.asarray(boHandle1.map())
    bo2 = np.asarray(boHandle2.map())
    bo3 = np.asarray(boHandle3.map())

    bo1[:] = wgt.reshape(-1).view(np.uint8)
    bo2[:] = inp.reshape(-1).view(np.uint8)
    boHandle1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, wgt.nbytes, 0)
    boHandle2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, inp.nbytes, 0)

    print("Kernel Run...")
    start = time.time()
    for i in range(args.iter):
        # run_hdl = dma(boHandle1, boHandle2, boHandle3, 
        #               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
        #               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
        #               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
        #               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
        #               TX, TY, TZ)
        run_hdl = dma(boHandle1, boHandle2, boHandle3, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, TX, TY, TZ, -1)
        state = run_hdl.wait()
    end = time.time()
    GOPs = M*K*N*2*args.iter*1e-9/(end-start)
    print("MM Size: {} x {} x {}".format(M, K, N))
    print("Total time is: {}s, OPs = {} GOPs".format((end-start), GOPs))

    boHandle3.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, res.nbytes, 0)
    bo3 = bo3.view(np.float32).reshape(N,M)
   
    if(args.verify == 1):
        if not np.allclose(res, bo3):
            print("Error....")
            diff = res-bo3
            print("Min diff = {}, Max diff = {}".format(diff.min(), diff.max()))
        else:
            print("Passed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python3 main.py -x <.xclbin> -m <M> -k <K> -n <N> -i <iter> -v <verify 0/1>')
    
    parser.add_argument('-x', '--xclbin', type=str, help='Path to xclbin', default='mm_hw.xclbin')
    parser.add_argument('-m', type=int, help='M', default=4096)
    parser.add_argument('-k', type=int, help='K', default=4096)
    parser.add_argument('-n', type=int, help='N', default=4096)
    parser.add_argument('-i', '--iter',type=int, help='iter', default=500)
    parser.add_argument('-v', '--verify', type=int, help='Enable verify', default=0)
    args = parser.parse_args()
    main(args)



