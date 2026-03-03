#!/bin/bash

module load singularity

OUTDIR=/network/projects/transdeconv/simu_lofar
SIF=/network/projects/transdeconv/apcon/OSKAR-WSClean.sif
LOFARGRID_DIR="${OUTDIR}/lofargrid"

for JOBTAG in MFS original; do
                    
                    LOFAR_OUTPUT="${LOFARGRID_DIR}/dirty_image_${JOBTAG}.fits"

                    if [ -f "${LOFAR_OUTPUT}" ]; then
                        continue
                    fi
                    
                    for GRIDDIR in "${LOFARGRID_DIR}" "${JVLABGRID_DIR}" "${JVLAAGRID_DIR}"; do
                        rm -f \
                            "${GRIDDIR}/continuum_${JOBTAG}.npz" \
                            "${GRIDDIR}/uvw_gridded_${JOBTAG}.npz" \
                            "${GRIDDIR}/dirty_image_${JOBTAG}.jpg" \
                            "${GRIDDIR}/dirty_image_${JOBTAG}.fits"
                    done
                    
                    rm -f script.sh
                    echo "#!/bin/bash" >> script.sh
                    echo "#SBATCH --job-name=lofar_${JOBTAG}" >> script.sh
                    echo "#SBATCH --output=${OUTDIR}/${JOBTAG}.txt" >> script.sh
                    echo "#SBATCH --error=${OUTDIR}/${JOBTAG}_error.txt" >> script.sh
                    echo "#SBATCH --ntasks=1" >> script.sh
                    echo "#SBATCH --time=1-00:00:00" >> script.sh
                    echo "#SBATCH --partition=main,main-cpu,long-cpu" >> script.sh
                    echo "#SBATCH --mem=32Gb" >> script.sh
                    echo "" >> script.sh
                    echo "cd ${OUTDIR}" >> script.sh
                    echo "" >> script.sh

                    echo "singularity exec --cleanenv \
    --bind /network \
    ${SIF} bash -c \"
        python3 -m pip install --user --break-system-packages \
            scipy python-casacore astropy matplotlib viscube==0.3.1 && \
        export PYTHONPATH=\$HOME/.local/lib/python3.12/site-packages:\$PYTHONPATH && \
        cd ../simu_lofar && \
        python3 gridder_merged.py ${JOBTAG}
                        \"" >> script.sh
                    
                    chmod +x script.sh
                    sbatch ./script.sh
                                    
done
