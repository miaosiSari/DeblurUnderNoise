function refined = refine_kernel(kernel, adjust, lightpruning)
    if ~exist('adjust', 'var')
        adjust = true;
    end
    
    if adjust
       refined = adjust_psf_center(kernel);
    else
       refined = kernel;
    end
    maxkernel = max(refined(:));
    refined(refined < (maxkernel/10.0)) = 0;
    if lightpruning
        CC = bwconncomp(refined,4);
        for ii=1:CC.NumObjects
            currsum=sum(refined(CC.PixelIdxList{ii}));
            if currsum < .1
                 refined(CC.PixelIdxList{ii}) = 0.0;
            end
        end   
    else
        CC = bwconncomp(refined,4);
        currsum = zeros(CC.NumObjects, 1);
        for ii=1:CC.NumObjects
            currsum(ii)=sum(refined(CC.PixelIdxList{ii}));
        end
        [~, maxid] = max(currsum);
        for ii=1:CC.NumObjects
            if ii ~= maxid
                refined(CC.PixelIdxList{ii}) = 0.0; 
            end
        end
    end
    refined = refined/sum(refined(:));
end