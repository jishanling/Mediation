% -------------------------------------------------------------------------
% Local function: bootstrap indirect effect
% -------------------------------------------------------------------------
function indirect_boot = bootstrapIndirectEffect(X, Y, M, Cov, n_boot)
    % Resamples data with replacement and computes a*b for each bootstrap sample
    n = length(X);
    indirect_boot = zeros(n_boot, 1);
    
    for k = 1:n_boot
        idx = randsample(n, n, true);
        X_bs = X(idx);
        M_bs = M(idx);
        Y_bs = Y(idx);
        C_bs = Cov(idx, :);
        
        % Model 2 (bootstrap): M ~ X + covariates
        lm2_bs = fitlm([X_bs, C_bs], M_bs);
        a_bs = lm2_bs.Coefficients.Estimate(2);
        
        % Model 3 (bootstrap): Y ~ X + M + covariates
        lm3_bs = fitlm([X_bs, M_bs, C_bs], Y_bs);
        b_bs = lm3_bs.Coefficients.Estimate(3);
        
        indirect_boot(k) = a_bs * b_bs;
    end
end