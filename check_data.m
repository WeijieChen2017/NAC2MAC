clear all ; close all

datadir='.';

for i=0:200

	currCT  = sprintf('%s%s%s%sdeepMRAC_direct%03.0f_%s.nii',datadir,filesep,'ct',filesep,i,'CTAC');
	currNAC  = sprintf('%s%s%s%sdeepMRAC_direct%03.0f_%s.nii',datadir,filesep,'nac',filesep,i,'NAC');
	currMAC  = sprintf('%s%s%s%sdeepMRAC_direct%03.0f_%s.nii',datadir,filesep,'mac',filesep,i,'MAC');
	currTOFNAC  = sprintf('%s%s%s%sdeepMRAC_direct%03.0f_%s.nii',datadir,filesep,'tofnac',filesep,i,'TOFNAC_R');

	check = 0;
	if isfile(currCT)
		check = check+1;
	end
	if isfile(currNAC)
		check = check + 1;
	end
	if isfile(currMAC)
		check = check + 1;
	end
	if isfile(currTOFNAC)
		check = check + 1;
	end

	if check~=4
		% not a good combo
		if isfile(currCT)
			movefile(currCT,[currCT '.nomatch']);
		end
		if isfile(currNAC)
			movefile(currNAC,[currNAC '.nomatch']);
		end
		if isfile(currMAC)
			movefile(currMAC,[currMAC '.nomatch']);
		end
		if isfile(currTOFNAC)
			movefile(currTOFNAC,[currTOFNAC '.nomatch']);
		end

		%fprintf('blah\n')
	end
end
