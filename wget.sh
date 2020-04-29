#!/bin/bash

dir="../data/plk/ysz/pub/"

check_file(){
	FILE=${dir}${1}
	if [ ! -f "${FILE}" ]; then
		wget -O ${FILE} "${2}"
	else
		echo "file exist: ${FILE}"
	fi
}

# //// DR2 //// #
# MILCA ymap 
check_file COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits

# NILC ymap
check_file COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits

# MASK
check_file COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits

# Full DR2 tar ball
check_file COM_CompMap_Compton-SZMap_R2.01.tgz https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Compton-SZMap_R2.01.tgz

# //// DR3 //// #

# Full DR3 tar ball
check_file COM_CompMap_Compton-SZMap_R2.02.tgz https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Compton-SZMap_R2.02.tgz


