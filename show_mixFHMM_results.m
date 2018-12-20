function show_mixFHMM_results(data,mixFHMM)
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaultaxesfontsize',14);


% symboles = {'b--o','b:s','b->','b.-x', '+', 'o', '*' ,'p', 'h' ,'o', '<', '^' ,'v'};
% couleurs = {'b--o', 'r:s','g->','m-s','black-s'};
% %
[n, m]=size(data);

t = 0:m-1;
scrsz = get(0,'ScreenSize');
figure('Position',[10 scrsz(4)/2 550 scrsz(4)/2.15]);
plot(t,data')
xlabel('t')
ylabel('y(t)')
title('original time series')

K = length(mixFHMM.param.w_k);

colors = {'r','g','b','k','m','y','c','r','g','b','k','m','y','c'};
figure('Position',[scrsz(4) scrsz(4)/2 550 scrsz(4)/2.15]);

for k=1:K
    plot(t,data(mixFHMM.stats.klas==k,:)','color',colors{k})
    % hold on, plot(t,solution.smoothed(:,k),'color',colors{K+k},'linewidth', 2.5);
    hold on
end
xlabel('t')
ylabel('y(t)')
title('Clustered time series')

K = size(mixFHMM.stats.tau_ik,2);
for k=1:K
    %subplot(G,1,g),plot(solution.tau_ijgk(:,:,g))
    %title(['\tau_{ ijgk} , g = ',int2str(g)])
    %     hold on, subplot(G,1,g),
    figure,plot(t,data(mixFHMM.stats.klas==k,:)','color',colors{k})
    hold on, plot(t,mixFHMM.stats.smoothed(:,k),'color',colors{K+k},'linewidth',2.5);
    xlabel('t')
ylabel('y(t)')
title(['Cluster ',int2str(k)])
end


