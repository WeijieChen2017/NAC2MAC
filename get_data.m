clear all ; close all

indir='../directSortedNii';
outdir='.';

for i=0:127
%for i=128:200
	currCT  = [indir filesep sprintf('deepMRAC_direct%03.0f_CTAC.nii',i)];
	currNAC = [indir filesep sprintf('deepMRAC_direct%03.0f_NAC.nii',i)];
	currMAC = [indir filesep sprintf('deepMRAC_direct%03.0f_MAC.nii',i)];

	if isfile(currCT)
		cmdline = sprintf('3dresample -prefix %s%sct%sdeepMRAC_direct%03.0f_CTAC.nii -dxyz 3 3 3 -rmode Cu -inset %s',outdir,filesep,filesep,i,currCT);
		system(cmdline)
	end

        if isfile(currNAC)
		cmdline = sprintf('3dresample -prefix %s%snac%sdeepMRAC_direct%03.0f_NAC.nii -dxyz 3 3 3 -rmode Cu -inset %s',outdir,filesep,filesep,i,currNAC);
                system(cmdline)
        end

        if isfile(currCT)
		cmdline = sprintf('3dresample -prefix %s%smac%sdeepMRAC_direct%03.0f_MAC.nii -dxyz 3 3 3 -rmode Cu -inset %s',outdir,filesep,filesep,i,currMAC);
                system(cmdline)
        end
	

end
