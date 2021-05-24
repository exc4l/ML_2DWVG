basep = cd;
datadirs = ["cycle_ndiff", "rand_ndiff", "unif_ndiff"];


for dp =datadirs
    filep = basep + "\" + dp;
for i=[0.1:0.1:4]
    coofp = filep+sprintf("\\%.1fcoo.csv", i);
    valfp = filep+sprintf("\\%.1fval.csv", i);
    maxvalfp = filep+sprintf("\\%.1fmaxval.csv", i);
    if ~isfile(coofp)
        i;
    else
        close all
        X = readmatrix(coofp);
        y = X(:,2);
        x = X(:,1);
        z = readmatrix(valfp);
        xv = linspace(min(x), max(x), 500);
        yv = linspace(min(y), max(y), 500);
        [X,Y] = meshgrid(xv, yv);
        Z = griddata(x,y,z,X,Y);
        figure(2)
        p = contourf(X, Y, Z);
        colorbar
        grid off
        shading interp
        
        colorDepth = 20;
        colormap(parula(colorDepth));
        view(2)
        title(sprintf("mae ndiff=%.1f", i));
        xlabel("height")
        ylabel("width")
        exportgraphics(gcf, filep+sprintf("\\mae_ndiff%.1f.png", i),'Resolution',300)
        close all
        X = readmatrix(coofp);
        y = X(:,2);
        x = X(:,1);
        z = readmatrix(maxvalfp);
        xv = linspace(min(x), max(x), 500);
        yv = linspace(min(y), max(y), 500);
        [X,Y] = meshgrid(xv, yv);
        Z = griddata(x,y,z,X,Y);
        figure(2)
        p = contourf(X, Y, Z);
%         hold on
%         contour(X, Y, Z);
%         % p.FaceAlpha = 0.3;
        colorbar
        grid off
        shading interp
        
        colorDepth = 20;
        colormap(parula(colorDepth));
        view(2)
        title(sprintf("maxerr ndiff=%.1f", i));
        xlabel("height")
        ylabel("width")
        exportgraphics(gcf, filep+sprintf("\\maxerr_ndiff%.1f.png", i),'Resolution',300)
    end
end
close all
errfp = filep+sprintf("\\errormetrics.csv");
errmat= readmatrix(errfp);
x = errmat(:,2);
hold on
plot(x, errmat(:,5),'DisplayName','MAE')
plot(x, errmat(:,6),'DisplayName','MSE')
set(gca, 'YScale', 'log')
legend('Location','northwest')
title("mse and mae vs ndiff")
exportgraphics(gcf,filep+"\\mse_mae_ndiff.png" ,'Resolution',300)
end
close all
figure(2)
set(gca, 'YScale', 'log')
hold on
for dp =datadirs
    filep = basep + "\" + dp;
    errfp = filep+sprintf("\\errormetrics.csv");
    errmat= readmatrix(errfp);
    x = errmat(:,2);
    plot(x, errmat(:,5),'DisplayName',dp+' MAE')
    plot(x, errmat(:,6),'DisplayName',dp+' MSE')
end
l = legend('Location','northwest')
set(l, 'Interpreter', 'none')
title("mse and mae vs ndiff")
exportgraphics(gcf,basep+"\\mse_mae_ndiff.png" ,'Resolution',300)