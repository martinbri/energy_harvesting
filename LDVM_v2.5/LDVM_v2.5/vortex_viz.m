clc;clear all;close all
vortex_data=load('flow_pr_amp45_k0.2_le.dat');

%Each time step is until an NaN is reached
start_ind=find(isnan(vortex_data(:,1)));
n_steps=length(start_ind)-1;
%%

figure
set(gcf,'Units','Inches','Position',[3 5 6 6]);
set(gcf,'DefaultAxesFontName','Helvetica');
set(gcf,'DefaultTextFontName','Helvetica');
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultTextFontSize',10);

set(gcf,'PaperUnits',get(gcf,'Units'));
pos = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 pos(3) pos(4)]);

set(gca,'Units','Inches');
set(gca,'Position',[0.5 0.5 5.0 5.0]);


vidobj=VideoWriter('flow_test.avi');
set(vidobj,'Quality',100);
vidobj.FrameRate=50
open(vidobj);
h=scatter(0,0,2.5,0,'filled');
%h2=line([0 0],[0 0],'color','k','Linewidth',2)
colormap(flipud(jet));
hold on;
axis equal;
axis([-10 1.2 -2.4 2.4])

%1.0082
colorbar;
caxis([-0.01 0.01])
for step=1:n_steps
    
    vort=vortex_data(start_ind(step)+1:start_ind(step+1)-1,:);
    n_vort=size(vort(:,1));
    n_vort=n_vort(:,1);
    range=1:n_vort;
    set(h,'Xdata',vort(range,2),'Ydata',vort(range,3),'Cdata',vort(range,1));
    %lex=vort(n_vort-198,2);
    %ley=vort(n_vort-198,3);
    %tex=vort(n_vort,2);
    %tey=vort(n_vort,3);
    %set(h2,'Xdata',[lex tex],'Ydata',[ley tey]);
     % pause(0.01);
    currframe=getframe;
    writeVideo(vidobj,currframe);
end
%print -depsc -loose theo_45_h2.eps
% cn_step=251;
% 
% for i=1:n_step
%     scatter

close(vidobj)

