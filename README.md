# ViT on Versal

## Install xtensor-xtl, xtensor, and xtensor-blas

```
cd 3rdParty/
git clone https://github.com/xtensor-stack/xtl
cd xtl 
mkdir build
cd build
cmake ..
sudo make install
cd ../../

git clone https://github.com/xtensor-stack/xtensor
cd xtensor 
mkdir build
cd build
cmake ..
sudo make install
cd ../../

git clone https://github.com/xtensor-stack/xtensor-blas
cd xtensor-blas 
mkdir build
cd build
cmake ..
sudo make install

```

## Build Instructions (VCK190)

```
source setup_vck190.sh
cd build_vck190
make all PLATFORM=${PLATFORM} SYSROOT=${SYSROOT} EDGE_COMMON_SW=${EDGE_COMMON_SW}
 ./hostexe mm_hw_xclbin <no. of iterations> # on VCK190's shell

```

## Build Instructions (VCK5000)

```
source setup_vck5000.sh
cd build_vck5000
make all
./hostexe mm_hw.xclbin
```


## Measuring Power on VCK190

1. Burn the sd card with the system controller image (BEAM Tool) as given [here](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/972914749/BEAM+Tool+for+VMK180+Evaluation+Kit#Board-Setup-and-Connection)

2. Connect the LAN to the single LAN port (not the stacked one) on the VCK190 to the internet

3. Before powering on the board connect the serial cable to the PC as well.

4. When powering on the board you will see 4 serial connections `ttyUSB<0-3>`, System Controller is at `ttyUSB3` and the versal itself is at `ttyUSB1`

5. Connect to the `ttyUSB3` (e.g. you can use `screen` tool, by giving the following command `sudo screen /dev/ttyUSB3 115200`).

6. After booting the system controller will tell you the IP address on which the BEAM tool is running.

7. Go to the browser and paste the ip along with the port i.e. `<ip address>:50002`

8. On the BEAM tool page go to the "Run Demos & Designs" -> "Versal Power Tool Run"

9. Go to the folder power-advantage-tool and open the `ipynb` file.

10. Now you can used the following simplified code to get the power values:

Note: If the website based tool does not work you can use ssh as well: `ssh root@<ip address>` password is `root`. And run the given below code with python3.


```
import time
import numpy as np
from poweradvantage import poweradvantage

TIME_IN_SEC = 40
RESOLUTION = 10 # number of points per second

pa = poweradvantage("VCK190", "SC")
max_power = 0
i = 0
p_list = []
while i<TIME_IN_SEC*RESOLUTION:
    total,power = pa.printpower()
    p_list.append(total)
    time.sleep(1./RESOLUTION)
    i += 1
p_list = np.array(p_list)
np.savetxt("power.log", p_list)

```


# Optional Content

## Adding pyxrt to your conda env (VCK5000)

For python3.8 only

```
conda activate
conda-develop /opt/xilinx/xrt/python
```

## Installing any package on VCK190

```
ssh root@<ip address of VCK190> (password is "root")
dnf install <package name>
```

## Installing PyTorch on VCK190

```
python3 -m ensurepip --default-pip
wget https://files.pythonhosted.org/packages/90/f6/b0358e90e10306f80c474379ae1c637760848903033401d3e662563f83a3/torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl
python3 -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade-pip
python3 -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl
```