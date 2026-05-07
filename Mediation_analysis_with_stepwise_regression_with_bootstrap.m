%% Mediation analysis with stepwise regression and bootstrap confidence intervals
%
% This script tests whether the relationship between an independent variable
% (air pollution, X) and multiple cognitive outcomes (Y) is mediated by
% functional network connectivity measures (M).
%
% Dependencies:
%   - Statistics and Machine Learning Toolbox (fitlm, randsample, prctile)
%   - Data matrices: air, CognitiveT, net1, Renkou (covariates)
 
% clear; clc;

% -------------------------------------------------------------------------
% 0. User-defined parameters
% -------------------------------------------------------------------------
alpha          = 0.05;        % significance level
n_bootstrap    = 5000;        % number of bootstrap samples
n_cog_tests    = 10;          % number of cognitive outcome variables
n_net_nodes    = 28;          % number of network nodes (mediator candidates)
covariate_cols = 1:3;         % columns of Renkou to be used as covariates

% -------------------------------------------------------------------------
% 1. Load and prepare data (replace with your actual data loading)
% -------------------------------------------------------------------------
% Assuming air, CognitiveT, net1, Renkou are already in the workspace
X_raw    = air(:, 1);                 % independent variable
Y_raw    = CognitiveT(:, 1:n_cog_tests); % all cognitive outcomes
M_raw    = net1(:, 1:n_net_nodes);       % all potential mediators
Cov_raw  = Renkou(:, covariate_cols);    % covariates

% Normalise continuous variables once (z-score)
X   = normalize(X_raw);
Cov = normalize(Cov_raw);

% -------------------------------------------------------------------------
% 2. Initialise storage for results
% -------------------------------------------------------------------------
% We will store all significant mediation paths in a structure array
Results = struct([]);

% Counters
cnt_significant_stepwise = 0;   % stepwise criterion (all three p < 0.05)
cnt_significant_bootstrap = 0;  % bootstrap CI excludes zero
cnt_nonsignificant_bootstrap = 0;

% Optional: also keep the original matrix format for backward compatibility
mediation_ofnet28 = [];
beta_NET28        = [];

% -------------------------------------------------------------------------
% 3. Main loops over cognitive tests and mediators
% -------------------------------------------------------------------------
for i = 1:n_cog_tests
    Y = normalize(Y_raw(:, i));    % normalise each cognitive outcome
    
    for j = 1:n_net_nodes
        M = normalize(M_raw(:, j));
        
        % =================================================================
        % Step 1: Stepwise regression (classic Baron & Kenny approach)
        % =================================================================
        n = length(X);
        
        % Design matrices (include covariates)
        X_design  = [ones(n,1), X, Cov];
        XM_design = [ones(n,1), X, M, Cov];
        
        % Model 1: Y ~ X + covariates   (total effect)
        lm1 = fitlm([X, Cov], Y);
        c_total = lm1.Coefficients.Estimate(2);   % coefficient of X
        p_c     = lm1.Coefficients.pValue(2);     % p-value for X
        
        % Model 2: M ~ X + covariates   (path a)
        lm2 = fitlm([X, Cov], M);
        a   = lm2.Coefficients.Estimate(2);
        p_a = lm2.Coefficients.pValue(2);
        
        % Model 3: Y ~ X + M + covariates   (path b and direct effect c')
        lm3 = fitlm([X, M, Cov], Y);
        c_prime = lm3.Coefficients.Estimate(2);   % direct effect of X
        b       = lm3.Coefficients.Estimate(3);   % effect of M
        p_cprime = lm3.Coefficients.pValue(2);
        p_b      = lm3.Coefficients.pValue(3);
        
        % Indirect effect
        indirect_effect = a * b;
        total_effect    = c_total;
        direct_effect   = c_prime;
        
        % Display stepwise results
        fprintf('Cognitive test %d, Mediator %d:\n', i, j);
        fprintf('  Total effect (c)     = %.4f, p = %.4f\n', c_total, p_c);
        fprintf('  Path a (X->M)        = %.4f, p = %.4f\n', a, p_a);
        fprintf('  Path b (M->Y)        = %.4f, p = %.4f\n', b, p_b);
        fprintf('  Direct effect (c'')  = %.4f, p = %.4f\n', c_prime, p_cprime);
        fprintf('  Indirect effect (ab) = %.4f\n', indirect_effect);
        fprintf('  Total = Direct + Indirect: %.4f = %.4f + %.4f\n', ...
                total_effect, direct_effect, indirect_effect);
        
        % Store paths if the stepwise criteria are met
        if (p_c < alpha && p_a < alpha && p_b < alpha)
            cnt_significant_stepwise = cnt_significant_stepwise + 1;
            % Store in a matrix (as in original code)
            beta_NET28(cnt_significant_stepwise, :) = [
                c_total, p_c, a, p_a, b, p_b, c_prime, i, j];
        end
        
        % =================================================================
        % Step 2: Bootstrap confidence interval for the indirect effect
        % =================================================================
        indirect_boot = bootstrapIndirectEffect(X, Y, M, Cov, n_bootstrap);
        CI = prctile(indirect_boot, [2.5, 97.5]);
        
        fprintf('  Bootstrap 95%% CI for indirect effect: [%.4f, %.4f]\n', CI);
        
        % Test whether the CI excludes zero
        if (CI(1) * CI(2) > 0)
            cnt_significant_bootstrap = cnt_significant_bootstrap + 1;
            fprintf('  -> Mediation is significant (p < 0.05, two-tailed)\n\n');
            
            % Store in a structure for easy handling
            Results(cnt_significant_bootstrap).cognitive_test    = i;
            Results(cnt_significant_bootstrap).mediator_node     = j;
            Results(cnt_significant_bootstrap).total_effect      = total_effect;
            Results(cnt_significant_bootstrap).p_total           = p_c;
            Results(cnt_significant_bootstrap).path_a            = a;
            Results(cnt_significant_bootstrap).p_a               = p_a;
            Results(cnt_significant_bootstrap).path_b            = b;
            Results(cnt_significant_bootstrap).p_b               = p_b;
            Results(cnt_significant_bootstrap).direct_effect     = c_prime;
            Results(cnt_significant_bootstrap).p_direct          = p_cprime;
            Results(cnt_significant_bootstrap).indirect_effect   = indirect_effect;
            Results(cnt_significant_bootstrap).CI_lower          = CI(1);
            Results(cnt_significant_bootstrap).CI_upper          = CI(2);
            
            % Also fill the old matrix, if needed
            mediation_ofnet28(cnt_significant_bootstrap, :) = [
                i, j, total_effect, p_c, a, p_a, b, p_b, ...
                c_prime, p_cprime, indirect_effect, CI(1), CI(2)];
        else
            cnt_nonsignificant_bootstrap = cnt_nonsignificant_bootstrap + 1;
            fprintf('  -> Mediation is not significant\n\n');
        end
        
        % Optional: plot bootstrap distribution
        % figure;
        % histogram(indirect_boot, 50);
        % hold on;
        % xline(0, 'r--', 'LineWidth', 2);
        % xline(CI(1), 'k--', 'LineWidth', 1.5);
        % xline(CI(2), 'k--', 'LineWidth', 1.5);
        % title(sprintf('Bootstrap: Cognitive %d - Node %d', i, j));
        % xlabel('Indirect effect');
        % ylabel('Frequency');
    end
end

% -------------------------------------------------------------------------
% Summary
% -------------------------------------------------------------------------
fprintf('========================================\n');
fprintf('Analysis complete.\n');
fprintf('Significant by stepwise criteria: %d\n', cnt_significant_stepwise);
fprintf('Significant by bootstrap CI:     %d\n', cnt_significant_bootstrap);
fprintf('Nonsignificant (bootstrap):      %d\n', cnt_nonsignificant_bootstrap);
fprintf('========================================\n'); 