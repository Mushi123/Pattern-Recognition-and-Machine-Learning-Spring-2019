%%
clc;
clear;
digits(5);
%rng('default');
rng(100);
%%
n = 100;
C = 100;
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
yi = [ones(60,1);-1*ones(40,1)];
%%
scatter(Xi(yi == 1,1),Xi(yi==1,2),80,'r','s','filled','DisplayName','Class 1','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
hold on;
scatter(Xi(yi == -1,1),Xi(yi==-1,2),50,'g','o','filled','DisplayName','Class 2','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);

%%
G = (yi*yi') .*( Xi*Xi');
e = ones (n ,1);
% l = eps ^.5;
% G = G + l*eye(n);
tic
ad = quadprog(G,-e,[],[],yi',0,zeros(n,1),C*ones(n,1));
ad = round(ad,5);
wqp = [0 0];
for i = 1:100
    wqp = wqp + ad(i)*yi(i)*Xi(i,:);
    disp(wqp);
end
wqp = wqp';
%%
% To find w0, find the support vectors. Support vectors have an epsilon
% value of 0 and a non-negative lambda which is less than C. Because if
% lambda is less than C, then mu has to be geater than 0 which means
% epsilon has to be 0. This has to signifiy the support vectors because in
% the other case where epsilon is 0 is when feature is not a support vector
% but in that case lambda is also 0

% For C = 0.1, features 9 and 89 have lamda not 0 but less than C so those
% are some of the support vecotrs. Both are in class 1. We use the eqn
% lambda_i[y_i(w.T.x_i + w0) - 1]=0. Here lambda_i is not 0 so the 2nd eqn
% has to be 0. We use examples 9 and 89 and take the avg value of the w0's 
% For example 9 the eqn is:
% [ 1(-0.6502*1.0971 + 0.6620*2.1159 + w0) - 1] = 0
% And get w0 to be around 0.3126
% For example 89, the eqn is
% [ -1(-0.6502*4.7716 + 0.6620*2.7037 + w0) - 1] = 0
% And get w0 = 0.3126
% We can then take avg of these two and get w0 = 0.0.3126

% We follow the same procedure for C = 100. In that case our support
% vectors are features 48, 61 and 72. w0 comes out to be 0.1958
%%
hold on
% [alpha , b, pos] = monqp (G,e,yi ,0,.1,l ,0) ;
% aqp = zeros (n ,1);
% aqp (pos) = alpha ;
% wqp = Xi(pos ,:)'*( yi(pos).* alpha );
%%
b = 0.3126; % For C = 0.1
if ( C == 100 )
    b = 0.1958; %1958
end  

sup_vecs = find(ad>0 & ad<C);
final = 0;
for j = 1:length(sup_vecs)
    final = final + (yi(sup_vecs(j)) - Xi(sup_vecs(j),:)*wqp);
end
w0 = final/length(sup_vecs);

b0 = -b/wqp(2);
b1 = -wqp(1)/wqp(2);
f = @(x) b0+b1*x;
fplot( f, [-3, 10],'black','DisplayName','Decision Boundary' )

hold on
b0 = (-b+1)/wqp(2);
b1 = -wqp(1)/wqp(2);
f = @(x) b0+b1*x;
fplot( f, [-3, 10],'--black','DisplayName','Upper Margin' )


b0 = (-b-1)/wqp(2);
b1 = -wqp(1)/wqp(2);
f = @(x) b0+b1*x;
fplot( f, [-3, 10],'--black','DisplayName','Lower Margin' )

hold on
%%
res  = yi.*[Xi*wqp+b];
res = round(res,5);
%%
scatter(Xi(( ad == C) & (ad > 0),1),Xi((ad == C) & (ad > 0),2),60,'x','b','LineWidth',1.5,'DisplayName','Violations');
hold on
scatter(Xi(( ad < C) & (ad > 0),1),Xi((ad < C) & (ad > 0),2),60,'.','b','LineWidth',2,'DisplayName','Margin');
hold on
scatter(Xi(res < 0,1),Xi(res < 0,2),100,'d','DisplayName','Misclassifications');
title(['C = ',num2str(C),' Sup. Vecs. = ',num2str(sum(ad > 0)),' Number of Misclassifications = ',num2str(length(Xi(res < 0,:)))]);
xlim([-2 8])
ylim([-2 6])
legend
hold off
%%
miss = length(Xi(res < 0,:)); % 9 for C = 0.1 and 7 for C = 100