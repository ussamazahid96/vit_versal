mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
TARGET   = hw
MODE	 = linux
HOST_ARCH = x86
Frequency = 250
PLATFORM_REPO_PATHS=/opt/xilinx/platforms
PLATFORM_NAME=xilinx_vck5000_gen4x8_xdma_2_202210_1
PLATFORM = ${PLATFORM_REPO_PATHS}/${PLATFORM_NAME}/${PLATFORM_NAME}.xpfm
XSA   = ${PLATFORM_NAME}_${TARGET}.xsa
HOST_EXE = hostexe
HOST_SRCS := ./host/host.cpp

FINAL_XCLBIN=mm_${TARGET}.xclbin
TEMP_DIR := ./_x.$(TARGET).${PLATFORM_NAME}
BUILD_DIR := ./build_dir.$(TARGET).${PLATFORM_NAME}

KERNEL_XO := $(TEMP_DIR)/dma.xo $(TEMP_DIR)/dma_small.xo $(TEMP_DIR)/layernorm.xo $(TEMP_DIR)/softmax.xo $(TEMP_DIR)/gelu.xo $(TEMP_DIR)/rant.xo
GRAPH   := aie/aie_top_all.cpp
LIBADF  = libadf.a
AIE_CMPL_CMD = aiecompiler -platform=${PLATFORM} --target=hw --stacksize=2048 --pl-register-threshold=250 --pl-freq=230  -include="./aie" -include="./aie/layer0" -include="./aie/layer1" -include="./aie/layer2" -include="./aie/layer3" -include="./aie/layer4" -include="./aie/layer5" -include="./aie/layer6" -include="./aie/layer7" -include="./aie/layer8" -workdir=./Work ${GRAPH}
AIE_SIM_CMD = aiesimulator --pkg-dir=./Work --dump-vcd foo --profile
AIE_HOSTEXE = aie_control_xrt.o

EMU_CMD = ./launch_hw_emu.sh
PACKAGE_OUT = ./package.$(TARGET)
RM = rm -f
RMDIR = rm -rf
##########################################################################################################################################################
### DO NOT MODIFY BELOW THIS LINE UNLESS NECESSARY
################################################################################################################################################

CUR_DIR := $(patsubst %/,%,$(dir $(mkfile_dir)))
PACKET_IDS_C_H :=./Work/temp/packet_ids_c.h
BINARY_CONTAINERS += $(BUILD_DIR)/${XSA}
VCC      = v++
VPP_XO_FLAGS += -c --platform $(PLATFORM) --save-temps -g --optimize 2
VPP_XO_FLAGS += --hls.jobs 8
VPP_XO_FLAGS += -I$(CUR_DIR)/kernel

CXX := g++
GCC_FLAGS := -Wall -c \
			 -std=c++17 \
			 -Wno-int-to-pointer-cast \
			 
GCC_INCLUDES := -I./  \
				-I${XILINX_VITIS}/aietools/include \
				-I${XILINX_VITIS}/include \
				-I/opt/xilinx/xrt/include \
				-I/usr/bin/boost_1_63_0

GCC_LIB := -lxrt_coreutil -lxrt_core -lxrt_coreutil \
		   -L$(XILINX_XRT)/lib \
		   -L${XILINX_VITIS}/aietools/lib/lnx64.o/



CLFLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps --optimize 2
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif
CLFLAGS += --hls.jobs 8
CLFLAGS += --config ./conn.cfg 
CLFLAGS += --kernel_frequency ${Frequency}

LDCLFLAGS := --vivado.synth.jobs 8 --vivado.impl.jobs 8

.PHONY: clean

###
# Guarding Checks. Do not modify.
###
check_defined = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))

__check_defined = \
	$(if $(value $1),, \
		$(error Undefined $1$(if $2, ($2))))

guard-PLATFORM_REPO_PATHS:
	$(call check_defined, PLATFORM_REPO_PATHS, Set your where you downloaded xilinx_vck190_es1_base_202110_1)



all: aie kernels build host package weights

weights:
	tar -xzvf ../PyTorch/vit.tar.gz

aie: guard-PLATFORM_REPO_PATHS ${LIBADF}
${LIBADF}: ${GRAPH}
	@rm -rf Work libadf.a
	@mkdir -p Work
	${AIE_CMPL_CMD}
	@echo "COMPLETE: aie success."

aiesim: ${LIBADF}
	${AIE_SIM_CMD}
	@echo "COMPLETE: aiesim success."


kernels:$(KERNEL_XO)
$(TEMP_DIR)/dma.xo: kernel/dma.cpp kernel/dma.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k dma --hls.clock 300000000:dma $< -o $@ | tee $(TEMP_DIR)/dma.log
	@echo "### ***** dma compilation done! *****"

$(TEMP_DIR)/dma_small.xo: kernel/dma_small.cpp kernel/dma_small.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k dma_small --hls.clock 300000000:dma_small $< -o $@ | tee $(TEMP_DIR)/dma_small.log
	@echo "### ***** dma_small compilation done! *****"

$(TEMP_DIR)/layernorm.xo: kernel/layernorm.cpp kernel/layernorm.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k layernorm --hls.clock 300000000:layernorm $< -o $@ | tee $(TEMP_DIR)/layernorm.log
	@echo "### ***** layernorm compilation done! *****"

$(TEMP_DIR)/softmax.xo: kernel/softmax.cpp kernel/softmax.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k softmax --hls.clock 220000000:softmax $< -o $@ | tee $(TEMP_DIR)/softmax.log
	@echo "### ***** softmax compilation done! *****"

$(TEMP_DIR)/gelu.xo: kernel/gelu.cpp kernel/gelu.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k gelu --hls.clock 300000000:gelu $< -o $@ | tee $(TEMP_DIR)/gelu.log
	@echo "### ***** gelu compilation done! *****"

$(TEMP_DIR)/rant.xo: kernel/rant.cpp kernel/rant.hpp
	mkdir -p $(TEMP_DIR)
	$(VCC) $(VPP_XO_FLAGS) -k rant --hls.clock 300000000:rant $< -o $@ | tee $(TEMP_DIR)/rant.log
	@echo "### ***** rant compilation done! *****"

build: $(BINARY_CONTAINERS)
$(BUILD_DIR)/${XSA}: ${KERNEL_XO} ${LIBADF}
	mkdir -p $(BUILD_DIR)
	v++ -l $(CLFLAGS) --temp_dir $(BUILD_DIR) $(LDCLFLAGS) -o $@ ${KERNEL_XO} ${LIBADF}

host:$(HOST_EXE)
$(HOST_EXE):
	$(CXX) $(HOST_SRCS) $(GCC_FLAGS) -fopenmp $(GCC_INCLUDES) -o main.o
	$(CXX) *.o $(GCC_LIB) -fopenmp -o $(HOST_EXE)
	@echo "COMPLETE: Host application created."

package:${FINAL_XCLBIN}
${FINAL_XCLBIN}: $(BINARY_CONTAINERS) $(LIBADF) 
	v++ -p -t $(TARGET) -f $(PLATFORM) $(BINARY_CONTAINERS) $(LIBADF) -o ${FINAL_XCLBIN} --package.boot_mode=ospi
	@echo "COMPLETE: package created."


clean:
	-$(RMDIR) $(HOST_EXE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv *.o *.xo *.xpe *.xsa cfg qemu_dts_files emu_qemu_scripts *.db sim  *.a  Work aie/*.ll _x*
	-$(RMDIR)  *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb *bin* *summary* *.BIN *.bif *.exe  *.log *.txt vit/

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.* run_app.sh 
	-$(RMDIR) *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

cleansim:
	-$(RMDIR) aiesimulator_output foo.vcd ISS_RPC_SERVER_PORT
cleantemp:
	-$(RMDIR) ./_x.$(TARGET).${PLATFORM_NAME}
cleanpac:
	-$(RMDIR) hostexe main.o mm_hw.xclbin boot_image.bif BOOT.BIN aie.merged.cdo.bin