function mixFHMM =  learn_MixFHMM(data, K, R, ...
    variance_type, order_constraint, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose)
%
% The EM algorithm for parameter estimation of the mixture of Hidden Markov
% Models for clustering and segmentation of time series with regime changes
%
% Inputs:
% data: a set of n time series with m observations (dim: [n x m]
% K: number of clusters
% R: number of regimes (states)
% options
%
%
%
% faicel chamroukhi (septembre 2009)
%
%% Please cite the following references for this code
%
% @InProceedings{Chamroukhi-IJCNN-2011,
%   author = {F. Chamroukhi and A. Sam\'e  and P. Aknin and G. Govaert},
%   title = {Model-based clustering with Hidden Markov Model regression for time series with regime changes},
%   Booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN), IEEE},
%   Pages = {2814--2821},
%   Adress = {San Jose, California, USA},
%   year = {2011},
%   month = {Jul-Aug},
%   url = {https://chamroukhi.com/papers/Chamroukhi-ijcnn-2011.pdf}
% }
%
% @PhdThesis{Chamroukhi_PhD_2010,
% author = {Chamroukhi, F.},
% title = {Hidden process regression for curve modeling, classification and tracking},
% school = {Universit\'e de Technologie de Compi\`egne},
% month = {13 december},
% year = {2010},
% type = {Ph.D. Thesis},
% url ={https://chamroukhi.com/papers/FChamroukhi-Thesis.pdf}
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off
[n, m] = size(data);%n  nbre de signaux (individus); m: nbre de points pour chaque signal
%
Y=reshape(data',[],1);

%
try_EM = 0;
best_loglik = -inf;
cputime_total = [];

while try_EM < total_EM_tries
    try_EM = try_EM +1;
    fprintf('EM try n° %d\n',try_EM);
    time = cputime;
    %%%%%%%%%%%%%%%%%%%
    %  Initialization %
    %%%%%%%%%%%%%%%%%%%
    mixFHMM = init_MixFHMM(data, K, R,...
        variance_type, order_constraint, init_kmeans, try_EM);
    
    %     Psi = zeros(nu,1);% vecteur parametre
    iter = 0;
    converge = 0;
    loglik = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    
    % main algorithm
    % % EM %%%%
    while ~converge && (iter< max_iter_EM)
        
        %
        exp_num_trans_ck  = zeros(R,R,n);
        exp_num_trans_from_l_ck = zeros(R,n);
        %
        exp_num_trans = zeros(R,R,n,K);
        exp_num_trans_from_l = zeros(R,n,K);
        %
        w_k_fyi = zeros(n,K);
        log_w_k_fyi = zeros(n,K);
        
        %%%%%%%%%%
        % E-Step %
        %%%%%%%%%%
        gamma_ikjr = zeros(n*m,R,K);
        for k=1:K
            % run a hmm for each sequence
            log_fkr_yij =zeros(R,m);
            %
            Li = zeros(n,1);% to store the loglik for each example
            %
            mu_kr = mixFHMM.param.mu_kr(:,k);
            
            for i=1:n
                y_i = data(i,:);
                for r = 1:R
                    mukr = mu_kr(r);
                    %sk = sigma_kr(k);
                    if strcmp(variance_type,'common')
                        sigma_kr = mixFHMM.param.sigma_k(k);
                        sk = sigma_kr;
                    else
                        sigma_kr = mixFHMM.param.sigma_kr(:,k);
                        sk = sigma_kr(r);
                    end
                    z=((y_i-mukr*ones(1,m)).^2)/sk;
                    log_fkr_yij(r,:) = -0.5*ones(1,m).*(log(2*pi)+log(sk)) - 0.5*z;% pdf cond à  c_i = g et z_i = k de yij
                    fkr_yij(r,:) = normpdf(y_i, mukr*ones(1,m), sqrt(sk));
                end
                %                         log_fkr_yij  = min(log_fkr_yij,log(realmax));
                %                         log_fkr_yij = max(log_fkr_yij ,log(realmin));
                %                         fkr_yij =  exp(log_fkr_yij);
                
                % calcul de p(y) : forwards backwards
                
                [gamma_ik, xi_ik, fwd_ik, backw_ik, loglik_i] = forwards_backwards(mixFHMM.param.pi_k(:,k), mixFHMM.param.A_k(:,:,k), fkr_yij);
                %
                
                
                Li(i) = loglik_i; % loglik of the ith curve
                %
                gamma_ikjr((i-1)*m+1:i*m,:,k) = gamma_ik';%[n*m K G]
                % xi_ikjrl(:,:,(i-1)*(m-1)+1:i*(m-1),g) =  xi_ik;%[KxK n*m G]
                %
                exp_num_trans_ck(:,:,i) = sum(xi_ik,3); % [K K n]
                exp_num_trans_from_l_ck(:,i) = gamma_ik(:,1);%[K x n]
                %
            end
            
            exp_num_trans_from_l(:,:,k) = exp_num_trans_from_l_ck;%[K n G]
            exp_num_trans(:,:,:,k) = exp_num_trans_ck;%[K K n G]
            
            % for the MAP partition:  the numerator of the cluster post
            % probabilities
            num_log_post_prob(:,k) = log(mixFHMM.param.w_k(k)) + Li;
            
            % for computing the global loglik
            w_k_fyi(:,k) = mixFHMM.param.w_k(k)*exp(Li);%[nx1]
            
            log_w_k_fyi(:,k) = log(mixFHMM.param.w_k(k)) + Li;
        end
        
        log_w_k_fyi = min(log_w_k_fyi,log(realmax));
        log_w_k_fyi = max(log_w_k_fyi,log(realmin));
        
        tau_ik = exp(log_w_k_fyi)./(sum(exp(log_w_k_fyi),2)*ones(1,K));
        
        % % log-likelihood
        loglik = sum(log(sum(exp(log_w_k_fyi),2)));
        
        
        %%%%%%%%%%
        % M-Step %
        %%%%%%%%%%
        
        % Maximization of Q1 w.r.t w_k
        mixFHMM.param.w_k = sum(tau_ik,1)'/n;
        for k=1:K
            
            if strcmp(variance_type,'common'), s=0; end
            
            weights_cluster_k = tau_ik(:,k);
            % Maximization of Q2 w.r.t \pi^g
            exp_num_trans_k_from_l =   (ones(R,1)*weights_cluster_k').*exp_num_trans_from_l(:,:,k);%[K x n]
            mixFHMM.param.pi_k(:,k) = (1/sum(tau_ik(:,k)))*sum(exp_num_trans_k_from_l,2);% sum over i
            % Maximization of Q3 w.r.t A^g
            for r=1:R
                if n==1
                    exp_num_trans_k(r,:,:) = (ones(R,1)*weights_cluster_k)'.*squeeze(exp_num_trans(r,:,:,k));
                else
                    %exp_num_trans_k(k,:,:,g)
                    exp_num_trans_k(r,:,:) = (ones(R,1)*weights_cluster_k').*squeeze(exp_num_trans(r,:,:,k));
                end
            end
            if n==1
                temp = exp_num_trans_k;
            else
                temp = sum(exp_num_trans_k,3);%sum over i
            end
            mixFHMM.param.A_k(:,:,k) = mk_stochastic(temp);
            % if HMM with order constraints
            if order_constraint
                mixFHMM.param.A_k(:,:,k) = mk_stochastic(mixFHMM.stats.mask.*mixFHMM.param.A_k(:,:,k));
            end
            
            % Maximisation de Q4 par rapport aux muk et sigmak
            % each sequence i (m observations) is first weighted by the cluster weights
            weights_cluster_k =  repmat((tau_ik(:,k))',m,1);
            weights_cluster_k = weights_cluster_k(:);
            
            % secondly, the m observations of each sequance are weighted by the
            % wights of each segment k (post prob of the segments for each
            % cluster g)
            gamma_ijk = gamma_ikjr(:,:,k);% [n*m K]
            
            nm_kr=sum(gamma_ijk,1);% cardinal nbr of the segments k,k=1,...,K within each cluster g, at iteration q
            
            sigma_kr = zeros(R,1);
            for r=1:R
                nmkr = nm_kr(r);%cardinal nbr of segment k for the cluster g
                % % Maximization w.r.t muk
                weights_seg_k = gamma_ijk(:,r);
                
                mu_kr(r) = (1/sum(weights_cluster_k.*weights_seg_k))*sum((weights_cluster_k.*weights_seg_k).*Y);
                % % Maximization w.r.t sigmak :
                z = sqrt(weights_cluster_k.*weights_seg_k).*(Y-ones(n*m,1)*mu_kr(r));
                if strcmp(variance_type,'common')
                    s = s + z'*z;
                    ngm = sum(sum((weights_cluster_k*ones(1,R)).*gamma_ijk));
                    sigma_k = s/ngm;
                else
                    ngmk = sum(weights_cluster_k.*weights_seg_k);
                    sigma_kr(r)=  z'*z/(ngmk);
                end
            end
            mixFHMM.param.mu_kr(:,k) = mu_kr;
            if strcmp(variance_type,'common')
                mixFHMM.param.sigma_k(k) = sigma_k;
            else
                mixFHMM.param.sigma_kr(:,k) = sigma_kr;
            end
        end
        iter=iter+1;
        
        if prev_loglik-loglik > 1e-3, fprintf(1, '!!!!! EM log-lik is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);end
        if verbose
            fprintf(1,'EM : Iteration : %d log-likelihood : %f \n',  iter,loglik);
        end
        
        converge =  abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik = [stored_loglik loglik];
        
    end % end of EM  loop
    cputime_total = [cputime_total cputime-time];
    
    mixFHMM.param = mixFHMM.param;
    if strcmp(variance_type,'common')
        mixFHMM.stats.Psi = [mixFHMM.param.w_k(:); mixFHMM.param.A_k(:); mixFHMM.param.pi_k(:); mixFHMM.param.mu_kr(:); mixFHMM.param.sigma_k(:)];
    else
        mixFHMM.stats.Psi = [mixFHMM.param.w_k(:); mixFHMM.param.A_k(:); mixFHMM.param.pi_k(:); mixFHMM.param.mu_kr(:); mixFHMM.param.sigma_kr(:)];
    end
    mixFHMM.stats.tau_ik = tau_ik;
    mixFHMM.stats.gamma_ikjr = gamma_ikjr;
    mixFHMM.stats.loglik = loglik;
    mixFHMM.stats.stored_loglik = stored_loglik;
    mixFHMM.stats.log_w_k_fyi = log_w_k_fyi;
    
    if mixFHMM.stats.loglik > best_loglik
        best_loglik = mixFHMM.stats.loglik;
        best_mixFHMM.stats = mixFHMM.stats;
    end
    
    if try_EM>=1,  fprintf('log-lik at convergence: %f \n', mixFHMM.stats.loglik); end
    
end
mixFHMM.stats.loglik = best_loglik;
%
if try_EM>1,  fprintf('log-lik max: %f \n', mixFHMM.stats.loglik); end

mixFHMM.stats = best_mixFHMM.stats;
% Finding the curve partition by using the MAP rule
[klas, Cik] = MAP(mixFHMM.stats.tau_ik);% MAP partition of the n sequences
mixFHMM.stats.klas = klas;

% cas ou on prend la moyenne des gamma ijkr
smoothed = zeros(m,K);
mean_curves = zeros(m,R,K);
%mean_gamma_ijk = zeros(m,R,K);
for k=1:K
    weighted_segments = sum(gamma_ikjr(:,:,k).*(Y*ones(1,R)),2);
    %weighted_segments = sum(gamma_ikjr(:,:,g).*(ones(n*m,1)*mixFHMM.param.mu_kr(:,k)'),2);
    
    %
    weighted_segments = reshape(weighted_segments,m,n);
    weighted_clusters = (ones(m,1)*mixFHMM.stats.tau_ik(:,k)').* weighted_segments;
    smoothed(:,k) = (1/sum(mixFHMM.stats.tau_ik(:,k)))*sum(weighted_clusters,2);
end

mixFHMM.stats.smoothed = smoothed;
mixFHMM.stats.mean_curves = mean_curves;
%mixFHMM.stats.mean_gamma_ijk = mean_gamma_ijk;

mixFHMM.stats.cputime = mean(cputime_total);

% % Segmentation of each cluster using the MAP rule
% for k=1:K
%     [segments_k Zjk] = MAP(mixFHMM.stats.mean_gamma_ijk(:,:,k));%MAP segmentation of each cluster of sequences
%     mixFHMM.stats.segments(:,k) = segments_k;
% end

nu = length(mixFHMM.stats.Psi);
% BIC AIC et ICL*
mixFHMM.stats.BIC = mixFHMM.stats.loglik - (nu*log(n)/2);%n*m/2!
mixFHMM.stats.AIC = mixFHMM.stats.loglik - nu;
% ICL*
% Compute the comp-log-lik
cik_log_w_k_fyi = (Cik).*(mixFHMM.stats.log_w_k_fyi);
comp_loglik = sum(sum(cik_log_w_k_fyi,2));
mixFHMM.stats.ICL1 = comp_loglik - nu*log(n)/2;%n*m/2!


