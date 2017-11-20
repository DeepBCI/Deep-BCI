function lZ = lp_mvniw(class_id, training_data, mu_0, k_0, v_0, lambda_0) 
%
% function lZ = lp_mvniw(class_id, training_data, mu_0, k_0, v_0, lambda_0) 
%   computes the marginal likelihood of the data under the MVN IW prior
%   
%       class_id: a vector of compact class_ids
%       training_data: a dxn vector of n, d-dimensional datapoints
%       mu_0, k_0, v_0, lambda_0: MVN IW prior parameters
%
%   returns log(prod_{j=1}^{K^+} P(Y^{(j)}; H))


lZ = 0;
d = size(training_data,1);

if d>v_0
    error('v_0 must be equal to or larger than the dimension of the data')
end

K_plus = length(unique(class_id));

    for l = 1:K_plus
        % get the class ids of the points sitting at table l
        hits= class_id==l;
        % how many are sitting at table l?
        n = sum(hits);
        % get those points
        Y = training_data(:,hits);
        
        % check for a problem. the classid's passed into this routine
        % should be compact (i.e. no classid's with no associated data)
        if n~=0
            
            %calculate the mean of the points at this table
            mean_Y = mean(Y,2);
            
            % set up the variables according to pg. 87 of Gelman
            
            % not needed 
%             mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*mean_Y;
            k_n = k_0+n;
            v_n = v_0+n;
            
            % the following line replaces the 5 following for reasons of 
            % efficiency
            S = cov(Y')'*(n-1);
%             temp = Y - repmat(mean_Y,1,size(Y,2));
%             S = zeros(size(lambda_0));
%             for ii=1:size(Y,2)
%                 S = S + temp(:,ii)*temp(:,ii)';
%             end
            
            lambda_n = lambda_0 + S ...
                + k_0*n/(k_0+n)*(mean_Y-mu_0)*(mean_Y-mu_0)';
        else
            error('Should always have one element')
        end
        
        % now that we have done this we can compute the marginal
        % probability of the data at this table under the prior. the
        % ratio of normalization constants is not given in Gelman but can
        % found instead in the paper Eqn's ?
        
        % the following split between even and odd is not necessary, the
        % hope is that the latter case is more efficient (particularly in a
        % lower level language than matlab
        if mod(n,2)~=0
%             disp('n odd')

            ls = 0;
            for j=1:d
                ls = ls + gammaln((v_n+1-j)/2) - gammaln((v_0+1-j)/2);
            end
        else
            ls = 0;
            for j=1:d
                for ii=1:floor(n/2)
                    ls = ls + log((v_n+1-j)/2-ii);
                end
            end
        
        end

        lZ=lZ-n*d/2 * log(2*pi) + d/2 * (log(k_0) - log(k_n)) + ...
            d/2*(v_n-v_0) * log(2) + v_0/2 * log(det(lambda_0)) ...
            - v_n/2 * log(det(lambda_n)) +ls;

    end