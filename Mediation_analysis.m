% Assume existing data: X (independent variable), M (mediator), Y (dependent variable), all as column vectors
% Please replace the following variables with actual data
% X = ...;
% M = ...;
% Y = ...;


% Pre-normalize all data
X = normalize(air);
Y_all = normalize(Cognitive);
M_all = normalize(Node);
Covars = normalize(Renkou);


mediation_results = [];
beta_results = [];
nnn = 0;
ppp = 0;

% Set Bootstrap parameters
n_bootstrap = 5000;  % Increase number of bootstrap resamples for better accuracy
n = length(X);

% Loop through each cognitive domain
for i = 1:size(Y_all, 2)
    Y = Y_all(:, i);
    
    % Display progress
    fprintf('Analyzing cognitive domain %d/%d\n', i, size(Y_all, 2));
    
    % Loop through each brain network metric
    for j = 1:size(M_all, 2)
        M = M_all(:, j);
        
        % Display progress
        fprintf('  Analyzing network metric %d/%d\n', j, size(M_all, 2));
        
        %% Step 1: Stepwise regression analysis (for reference only)
        % Model 1: Y ~ X + Covariates (total effect)
        [beta1, ~, ~, ~, stats1] = regress(Y, [ones(n,1), X, Covars]);
        c_total = beta1(2);  % total effect
        p_c = stats1(3);

        % Model 2: M ~ X + Covariates
        [beta2, ~, ~, ~, stats2] = regress(M, [ones(n,1), X, Covars]);
        a = beta2(2);  % effect of X on M
        p_a = stats2(3);

        % Model 3: Y ~ X + M + Covariates
        [beta3, ~, ~, ~, stats3] = regress(Y, [ones(n,1), X, M, Covars]);
        c_prime = beta3(2);  % direct effect of X on Y
        b = beta3(3);        % effect of M on Y
        p_b = stats3(3);

        % Calculate indirect effect
        indirect_effect = a * b;
        
        % Store stepwise regression results (regardless of significance)
        ppp = ppp + 1;
        beta_results(ppp, 1) = c_total;    % total effect
        beta_results(ppp, 2) = p_c;
        beta_results(ppp, 3) = a;          % X->M path
        beta_results(ppp, 4) = p_a;
        beta_results(ppp, 5) = b;          % M->Y path
        beta_results(ppp, 6) = p_b;
        beta_results(ppp, 7) = c_prime;    % direct effect
        beta_results(ppp, 8) = i;          % cognitive domain index
        beta_results(ppp, 9) = j;          % network metric index
        beta_results(ppp, 10) = indirect_effect; % indirect effect

        %% Step 2: Bootstrap method (primary testing method)
        indirect_effects = zeros(n_bootstrap, 1);
        direct_effects = zeros(n_bootstrap, 1);
        
        for k = 1:n_bootstrap
            % Resampling with replacement (including covariates)
            indices = randsample(n, n, true);
            X_bs = X(indices);
            M_bs = M(indices);
            Y_bs = Y(indices);
            Covars_bs = Covars(indices, :);
            
            % Model 2 (bootstrap sample): M ~ X + Covariates
            beta2_bs = regress(M_bs, [ones(n,1), X_bs, Covars_bs]);
            a_bs = beta2_bs(2);
            
            % Model 3 (bootstrap sample): Y ~ X + M + Covariates
            beta3_bs = regress(Y_bs, [ones(n,1), X_bs, M_bs, Covars_bs]);
            b_bs = beta3_bs(3);
            c_prime_bs = beta3_bs(2);
            
            % Store indirect and direct effects
            indirect_effects(k) = a_bs * b_bs;
            direct_effects(k) = c_prime_bs;
        end
        
        % Calculate 95% confidence intervals
        CI_indirect = prctile(indirect_effects, [2.5, 97.5]);
        CI_direct = prctile(direct_effects, [2.5, 97.5]);
        
        % Calculate Bootstrap p-value
        p_indirect = 2 * min(...
            sum(indirect_effects > 0) / n_bootstrap, ...
            sum(indirect_effects < 0) / n_bootstrap ...
        );
        
        % Store Bootstrap results
        if (CI_indirect(1) * CI_indirect(2) > 0) % confidence interval does not contain 0
            nnn = nnn + 1;
            mediation_results(nnn, 1) = i;              % cognitive domain index
            mediation_results(nnn, 2) = j;              % network metric index
            mediation_results(nnn, 3) = c_total;        % total effect
            mediation_results(nnn, 4) = p_c;            % total effect p-value
            mediation_results(nnn, 5) = a;              % X->M path
            mediation_results(nnn, 6) = p_a;            % X->M p-value
            mediation_results(nnn, 7) = b;              % M->Y path
            mediation_results(nnn, 8) = p_b;            % M->Y p-value
            mediation_results(nnn, 9) = c_prime;        % direct effect
            mediation_results(nnn, 10) = stats3(3);     % direct effect p-value
            mediation_results(nnn, 11) = indirect_effect; % indirect effect
            mediation_results(nnn, 12) = CI_indirect(1); % indirect effect 95% CI lower bound
            mediation_results(nnn, 13) = CI_indirect(2); % indirect effect 95% CI upper bound
            mediation_results(nnn, 14) = p_indirect;    % Bootstrap p-value
            mediation_results(nnn, 15) = CI_direct(1);  % direct effect 95% CI lower bound
            mediation_results(nnn, 16) = CI_direct(2);  % direct effect 95% CI upper bound
            
            fprintf('  Mediation effect significant: Cognitive domain %d, Network metric %d\n', i, j);
        else
           
            fprintf('  Mediation effect not significant: Cognitive domain %d, Network metric %d\n', i, j);
        end
    end
end

% Output summary
fprintf('\nAnalysis complete!\n');
fprintf('Total mediation models tested: %d\n', ppp);
fprintf('Models with significant mediation effect: %d\n', nnn);
fprintf('Models with non-significant mediation effect: %d\n', m);

% Multiple comparison correction
if ~isempty(mediation_results)
    p_values = mediation_results(:, 14); % Bootstrap p-values
    [fdr_corrected_p, ~] = mafdr(p_values, 'BHFDR', true);
    mediation_results(:, 17) = fdr_corrected_p; % Add FDR-corrected p-values
    
    fprintf('After FDR correction, %d models remain significant (q<0.05)\n', sum(fdr_corrected_p < 0.05));
end
