function [GT_data] =  MTPC_generate_metrics_mjlee(CM, selec, thresh)

% Thresholding

for k = 1:size(CM,3) % number of subjects
    
    fprintf('Evaluting %u matrices... \n',k);
    
    for i = 1:length(thresh) % number of thresholds
        TempCM = CM(:,:,k);
        threCM = threshold_proportional(TempCM, thresh(1,i));
        %         threCM = threshold_absolute(TempCM, thresh(1,i));
        
        %         biCM = zeros(length(threCM),length(threCM));
        %         % binary graph
        %         for m = 1:length(threCM)
        %             for n = 1:length(threCM)
        %                 if threCM(m,n) > 0
        %                     biCM(m,n) = 1;
        %                 end
        %             end
        %         end
        
        
        % applying to graph theory
        
        if selec == 1
%             GT1(k,i,:) = clustering_coef_wu(threCM)';
%             GT2(k,i,:) = betweenness_wei(threCM)';
%             GT3(k,i,:) = modularity_und(threCM)';
%             GT4(k,i,:) = eigenvector_centrality_und(threCM)';
            
%             GT1(k,i,:) = efficiency_wei(threCM,1)';
%             GT2(k,i,:) = strengths_und(threCM)';
% %             
            D=distance_wei(1./threCM); GT1(k,i,:)=charpath(D)';
            GT2(k,i,:) = transitivity_wu(threCM)';
% %             GT5(k,i,:) = efficiency_wei(threCM)';
% %             GT6(k,i,:) = SmallWorld(threCM)';
            
            
            
        else
            GT1(k,i,:) = degrees_dir_id(threCM)';
            GT2(k,i,:) = degrees_dir_od(threCM)';
            
            %             GT1(k,i,:) = degrees_dir_id(biCM)';
            %             GT2(k,i,:) = degrees_dir_od(biCM)';
        end
    end
    
    if selec == 1
        
        GT_data{1,1} = GT1;
        GT_data{1,2} = GT2;
%         GT_data{1,3} = GT3;
%         GT_data{1,4} = GT4;
%         GT_data{1,5} = GT5;
%         GT_data{1,6} = GT6;
%         GT_data{1,7} = GT7;
%         GT_data{1,8} = GT8;
%         GT_data{1,9} = GT9;
%         GT_data{1,10} = GT10;
        
    else
        GT_data{1,1} = GT1;
        GT_data{1,2} = GT2;
    end
    
end

