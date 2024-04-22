#!/bin/bash


pgkyl 5m_wavy-ion_${1}.gkyl sel -c 0,1,2,4 ev 'f[0][3] f[0][1] 2 pow f[0][0] / f[0][2] 2 pow f[0][0] / + 2 / - 1.5 / f[0][0] /' plot
