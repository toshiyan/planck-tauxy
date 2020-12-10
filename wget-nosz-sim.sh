#!/bin/bash

dir="/global/cscratch1/sd/toshiyan/PR3/cmb/sim/"

check_file(){
	FILE=${dir}${1}
	if [ ! -f "${FILE}" ]; then
		wget -O ${FILE} "${2}"
	else
		echo "file exist: ${FILE}"
	fi
}

# //// DR3 //// #
# SMICA No-SZ
for id in {000..100}
do
    #check_file dx12_v3_smica_nosz_cmb_mc_00${id}_raw.fits "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=dx12_v3_smica_nosz_cmb_mc_00${id}_raw.fits"
    check_file dx12_v3_smica_nosz_noise_mc_00${id}_raw.fits "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=dx12_v3_smica_nosz_noise_mc_00${id}_raw.fits"
done

