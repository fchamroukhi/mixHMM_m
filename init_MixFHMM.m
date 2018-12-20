function mixFHMM = init_MixFHMM(Y, K, R, ...
    variance_type, order_constraint, init_kmeans, try_algo)
%
%
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%

[n, m]=size(Y);

% % 1. Initialization of cluster weights
mixFHMM.param.w_k=1/K*ones(K,1);
% Initialization of the model parameters for each cluster
if init_kmeans
    max_iter_kmeans = 400;
    n_tries_kmeans = 20;
    verbose_kmeans = 0;
    res_kmeans = myKmeans(Y, K, n_tries_kmeans, max_iter_kmeans,verbose_kmeans);
    for k=1:K
        Yk = Y(res_kmeans.klas==k ,:); %if kmeans
        
        mixFHMM_init =  init_gauss_hmm(Yk, R, order_constraint, variance_type, try_algo);
        
        % 2. Initialisation de \pi_k
        mixFHMM.param.pi_k(:,k) = mixFHMM_init.initial_prob;%[1;zeros(R-1,1)];
        
        % 3. Initialisation de la matrice des transitions
        mixFHMM.param.A_k(:,:,k)  =  mixFHMM_init.trans_mat;
        if order_constraint
            mixFHMM.stats.mask = mixFHMM_init.mask;
        end
        
        % 4. Initialisation des moyennes
        mixFHMM.param.mu_kr(:,k) = mixFHMM_init.mur;
        if strcmp(variance_type,'common')
            mixFHMM.param.sigma_k(k) = mixFHMM_init.sigma;
        else
            mixFHMM.param.sigma_kr(:,k) = mixFHMM_init.sigma2r;
        end
    end
else
    ind = randperm(n);
    for k=1:K
        if k<K
            Yk = Y(ind((k-1)*round(n/K) +1 : k*round(n/K)),:);
        else
            Yk = Y(ind((k-1)*round(n/K) +1 : end),:);
        end
        mixFHMM_init =  init_gauss_hmm(Yk, R, order_constraint, variance_type, try_algo);
        
        % 2. Initialisation de \pi_k
        mixFHMM.param.pi_k(:,k) = mixFHMM_init.initial_prob;%[1;zeros(R-1,1)];
        
        % 3. Initialisation de la matrice des transitions
        mixFHMM.param.A_k(:,:,k)  =  mixFHMM_init.trans_mat;
        if order_constraint
            mixFHMM.stats.mask = mixFHMM_init.mask;
        end
        % 4. Initialisation des moyennes
        mixFHMM.param.mu_kr(:,k) = mixFHMM_init.mur;
        if strcmp(variance_type,'common')
            mixFHMM.param.sigma_k(k) = mixFHMM_init.sigma;
        else
            mixFHMM.param.sigma_kr(:,k) = mixFHMM_init.sigma2r;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param =  init_gauss_hmm(Y, R, order_constraint, variance_type, try_EM)
% init_gauss_hmm  estime les paramètres initiaux d'un hmm où la loi conditionnelle des observations est une gaussienne
%
% Entrees :
%
%        Y(i,:,nsignal) = x(i) : observation à l'instant i du signal
%        (séquence) nsignal (notez que pour la partie parametrisation des
%        signaux les observations sont monodimentionnelles)
%        R : nbre d'états (classes) cachés
%
% Sorties :
%
%         model : parametres initiaux du modele. structure
%         contenant les champs: para: structrure with the fields:
%         * le HMM initial
%         1. initial_prob (k) = Pr(Z(1) = k) avec k=1,...,K. loi initiale de z.
%         2. trans_mat(\ell,k) = Pr(z(i)=k | z(i-1)=\ell) : matrice des transitions
%         *
%         3.1. mur : moyenne de l'état k
%         3.2 sigma2r(k) = variance de x(i) sachant z(i)=k; sigma2r(j) =
%         sigma^2_r.
%         mu(:,k) = Esperance de x(i) sachant z(i) = k ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if order_constraint
    % % Tnitialisation en tenant compte de la contrainte:
    
    % Initialisation de la matrice des transitions
    mask = eye(R);%mask d'ordre 1
    for r=1:R-1
        ind = find(mask(r,:) ~= 0);
        mask(r,ind+1) = 1;
    end
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = [1;zeros(R-1,1)];
    param.trans_mat = normalize(mask,2);%
    param.mask = mask;
else
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = 1/R*ones(R,1);
    param.trans_mat = mk_stochastic(rand(R,R));
end

%  Initialisation des moyennes et des variances.
param_gauss = init_gauss_param_hmm(Y, R, variance_type, try_EM);

param.mur = param_gauss.mur;
if strcmp(variance_type,'common')
    param.sigma = param_gauss.sigma;
else
    param.sigma2r = param_gauss.sigma2r;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = init_gauss_param_hmm(Y, R, variance_type, try_EM)
% init_regression_model estime les parametres de la loi conditionnelle
% des observations : une gaussienne d'un hmm homogène d'ordre 1
%
% Entrees :
%
%        Y : [nxm]
%        nsignal (notez que pour la partie parametrisation des signaux les
%        observations sont monodimentionnelles)
%        R : nbre d'états (classes) cachés
% Sorties :
%
%
%         para : parametres initiaux de la loi cond de chaque état
%         2. sigma2r(r) = variance de y(t) sachant z(t)=r; sigmar(j) =
%         sigma^2_r.
%         3. mu(:,r) : E[y(t)|z(t) =r] ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, m] = size(Y);

if strcmp(variance_type,'common'),  s=0; end

if (try_EM) ==1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %decoupage de l'echantillon (signal) en K segments
    zi = round(m/R)-1;
    for r=1:R
        i = (r-1)*zi+1;
        j = r*zi;
        
        Yij = Y(:,i:j);
        Yij = reshape(Yij',[],1);
        param.mur(r) = mean(Yij);
        if strcmp(variance_type,'common')
            s=s+ sum((Yij-param.mur(r)).^2);
            param.sigma = s/(n*m);
        else
            m_r = j-i+1 ;
            param.sigma2r(r) = sum((Yij-param.mur(r)).^2)/(n*m_r);
        end
    end
else % initialisation aléatoire
    Lmin= 2;%round(m/(K+1));%nbr pts min dans un segments
    tr_init = zeros(1,R+1);
    tr_init(1) = 0;
    R_1=R;
    for r = 2:R
        R_1 = R_1-1;
        temp = tr_init(r-1)+Lmin:m-R_1*Lmin;
        ind = randperm(length(temp));
        tr_init(r)= temp(ind(1));
    end
    tr_init(R+1) = m;
    for r=1:R
        i = tr_init(r)+1;
        j = tr_init(r+1);
        Yij = Y(:,i:j);
        Yij = reshape(Yij',[],1);
        
        param.mur(r) = mean(Yij);
        
        if strcmp(variance_type,'common')
            s=s+ sum((Yij-param.mur(r)).^2);
            param.sigma = s/(n*m);
        else
            m_r = j-i+1 ;
            param.sigma2r(r) = sum((Yij-param.mur(r)).^2)/(n*m_r);
        end
    end
end
