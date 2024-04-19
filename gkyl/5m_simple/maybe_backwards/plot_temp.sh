#!/bin/bash


pgkyl RTI_test-ion_${1}.gkyl sel -c 0,1,2,4,7 ev 'f[0][3] f[0][0] f[0][1] f[0][0] / 2 pow * - f[0][0] /' plot
