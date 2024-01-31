using Interpolations
using Plots
using Measures
using Lux
using DifferentialEquations
using DiffEqFlux
using Optimization
using OptimizationFlux
using Random
using ComponentArrays
using Optim
using OptimizationOptimJL

smthstepgen(x,middlestep,rnge,xmin,tcenter) = rnge*tanh(middlestep*(x-tcenter))/2 + rnge/2 + xmin

function multiplestepvaluesmthtestν(x1,x2,ts1,t1e,mdlstp,timestep)
	#ts11 = 0.2
	#ts12 = -0.2
	xt = zeros(size(timestep))

	xt[1:t1e] = [smthstepgen(x,mdlstp,x2-x1,x1,ts1) for x in timestep[1:t1e]]
	return xt
end

function GenSmthStepΒ()
	stepν1smth11 = multiplestepvaluesmthtestν(0.40,0.235,15,30,0.2,1:1:30)
	stepν1smth12 = multiplestepvaluesmthtestν(0.235,0.112,9,30,0.25,1:1:30)
	stepν1smth13 = multiplestepvaluesmthtestν(0.112,0.055,13,30,0.5,1:1:30)
	stepν1smth14 = multiplestepvaluesmthtestν(0.055,0.152,3,20,0.5,1:1:20)
	stepν1smth15 = multiplestepvaluesmthtestν(0.152,0.106,26,40,0.5,1:1:40)
	stepν1smth16 = multiplestepvaluesmthtestν(0.106,0.13,30,40,0.5,1:1:40)
	stepν1smth17 = multiplestepvaluesmthtestν(0.13,0.153,20,40,0.5,1:1:40)
	#stepνsmth4 = multiplestepvaluesmthtestν(1.4,1.7,5,10,0.25,1:1:10)
	#2ndwave
	stepν1smth18 = multiplestepvaluesmthtestν(0.153,0.14,30,40,0.8,1:1:40)
	stepν1smth19 = multiplestepvaluesmthtestν(0.14,0.3,18,30,0.5,1:1:30)
	stepν1smth110 = multiplestepvaluesmthtestν(0.3,0.21,4,10,0.5,1:1:10)
	stepν1smth111 = multiplestepvaluesmthtestν(0.21,0.36,12,30,0.5,1:1:30)
	stepν1smth112 = multiplestepvaluesmthtestν(0.36,0.54,18,30,0.5,1:1:30)
	stepν1smth113 = multiplestepvaluesmthtestν(0.54,1.0,5,20,0.5,1:1:20)
	stepν1smthend1 = multiplestepvaluesmthtestν(1.0,1.2,10,61,0.2,1:1:61)
	stepβsmth21 = vcat(stepν1smth11,stepν1smth12,stepν1smth13,stepν1smth14,stepν1smth15,stepν1smth16,stepν1smth17,stepν1smth18,stepν1smth19,stepν1smth110,stepν1smth111,stepν1smth112,stepν1smth113,stepν1smthend1)


	#pβtsmth = interpolate((xt,),stepβsmth, Gridded(Constant()));
	#pβtsmth2 = interpolate((xt,),stepβsmth21, Gridded(Constant()));
	
	#pγtsmth2 = interpolate((xt,),stepγsmth112, Gridded(Constant()));
    return stepβsmth21
end
stepβsmth21 = GenSmthStepΒ()
startdate = 0
SimRange = 450
xt = [x for x = 0:1:(SimRange)]
pβtsmth2 = interpolate((xt,),stepβsmth21, Gridded(Constant()))
#pβt = interpolate((xt,),stepβ, Gridded(Constant()));
plot(pβtsmth2)
plot!(stepβsmth21[1:200])
# us011 = [1.02 - (Idiffsumnorm[1] + Rsumnorm[1]); Idiffsumnorm[1];Rsumnorm[1];1.0]
DRange = 450


u02 = [1.02 - (1.8140235813995465e-6 + 1.4738941598871314e-5); 1.8140235813995465e-6;1.4738941598871314e-5]	
#update
tspans011 = (0.0,DRange)
ps_original011 = [2.0,1.0,0.01,0.01]
#pp2 = (bt) #(bt,p_original2)
pps011 = (pβtsmth2 ,ps_original011)
datasizes011 = 450 + 1;
ts011 = range(tspans011[1],tspans011[2],length=datasizes011)

function sirmodel1!(du,u,p,t)

	β1,p2 = p
	β,γ,μ,λ = p2
	#bt = p#bb = B1[t]#γ = 0.01#λ = 0.01#ll1 = λ1[t]
    du[1] = - β1(t) * u[2] * u[1]

    du[2] = (β1(t) * u[2] * u[1] - 0.09 .* u[2])

    du[3] = 0.09 .* u[2] #- μ*u[4]

    #u[4] = u[1] + u[2]  + u[3] #+ u[4]
end
probsirsmth011 = ODEProblem(sirmodel1!,u02,tspans011,pps011,saveat=ts011);
compartment_solsmth011 = solve(probsirsmth011,Tsit5());
result1 = Array(compartment_solsmth011)
simstart = 240; simend = 390
simstart = 1; simend = 400
scatter(result1[2,simstart:simend])


#NN part 

rng = Random.default_rng()
ann = Lux.Chain(Lux.Dense(2,10,relu),Lux.Dense(10,1))
ann = Lux.Chain(Lux.Dense(2, 12, tanh),Lux.Dense(12, 3, tanh), Lux.Dense(3 ,1))

p1, st1 = Lux.setup(rng, ann)


parameter_array = Float64[0.15, 0.09, 0.01]

p0_vec = (layer_1 = p1, layer_2 = parameter_array)
p0_vec = ComponentArray(p0_vec)

function NNSIR(du, u, p, t)
    #β = abs(p.layer_2[1])
    #γ = abs(p.layer_2[2])
    #δ = abs(p.layer_2[3])

    UDE_term = abs(ann([u[1]; u[2]], p.layer_1, st1)[1][1])

    du[1]=  - UDE_term*u[1]*(u[2]) #/u0[1]

    du[2] = UDE_term*u[1]*(u[2]) - 0.09 .* u[2] #- UDE_term*u[2]/u0[1]
    du[3] = 0.09 .* u[2]

end

α = p0_vec


result1[1,simstart]
result1[2,simstart]
result1[3,simstart]
#u0 = Float64[60000000.0, 593, 62, 10]
u0 = [0.99; 1.8140235813995465e-6;1.4738941598871314e-5]
u0 = [1.02 - (1.8140235813995465e-6 + 1.4738941598871314e-5); 1.8140235813995465e-6;1.4738941598871314e-5]
u0 = [result1[1,simstart],result1[2,simstart],result1[3,simstart]]
tspan = (0, simend - simstart)
datasize = simend - simstart + 1;

prob= ODEProblem{true}(NNSIR, u0, tspan)
t = range(tspan[1],tspan[2],length=datasize)

#assaaaa = Array(solve(prob,Tsit5(),p=p0_vec,saveat=t,
#                 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))

assaaaa = loss_adjoint(p0_vec)
#comp1 = predict_adjoint(p0_vec)
#losstest = sum(abs2, log.(abs.(result1[2,:])) .- log.(abs.(comp1[2, :] ))) + (sum(abs2, log.(abs.(result1[3,:]) ) .- log.(abs.(comp1[3, :] ))))
      


function predict_adjoint(θ) # Our 1-layer neural network
    x = Array(solve(prob,Tsit5(),p=θ,saveat=t,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end


function loss_adjoint(θ)
      prediction = predict_adjoint(θ)
      #loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4, :] ))) + (sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
      loss = sum(abs2, log.(abs.(result1[2,simstart:simend])) .- log.(abs.(prediction[2,: ] ))) + (sum(abs2, log.(abs.(result1[3,simstart:simend]) ) .- log.(abs.(prediction[3,:] ))))
      
      return loss
end



iter = 0
function callback(θ,l)
    global iter
    iter += 1
    if iter%100 == 0
    #println(l)
    println("current iter $iter at loss $l")
    end
    return false
end


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec) #α)
res1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 50000) #15000)

optprob2 = remake(optprob,u0 = res3.u)

res2 = Optimization.solve(optprob2,Optim.BFGS(initial_stepnorm=0.001),
                                        callback=callback,
                                        maxiters = 100)

#optprob3 = remake(optprob2,u0 = res3.u)
res3 = Optimization.solve(optprob2, ADAM(0.01), callback = callback, maxiters = 20000)

data_pred = predict_adjoint(res3.u)
p3n = res3.u

data_pred = predict_adjoint(res2.u)
p3n = res2.u
#plot(data_pred[1,1:end])
bar(result1[2,simstart:simend],label="Model I",color="Grey",alpha=0.3,xlabel="Days",ylabel="Cases", title = "Optimization Plot")
plot!(data_pred[2,1:end],label="NN Predict I Group")

bar(result1[3,1:450],alpha=0.1)
plot!(data_pred[3,1:end])

bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, data_pred[2, :] .+ data_pred[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
plot!(t, data_pred[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Recovered + Dead)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

S_NN_all_loss = data_pred[1, :]
I_NN_all_loss = data_pred[2, :]
R_NN_all_loss = data_pred[3, :]
#T_NN_all_loss = data_pred[4, :]

Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
    #UDE_term = abs(ann([u[1]; u[2]; u[3]], p.layer_1, st1)[1][1])
    #abs(ann([u[1]; u[2]; u[3]], p.layer_1, st1)[1][1])
    #Q_parameter[i] = abs(re(p3n[1:51])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i]])[1])
  Q_parameter[i] = abs(ann([S_NN_all_loss[i]; I_NN_all_loss[i]], p3n.layer_1, st1)[1][1]) #ann([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i]])[1])
end


bb = vec(Q_parameter)
#stepβ
plot(Q_parameter[1:simend-simstart],label="NN Predict β(t)",ylabel="β(t)",xlabel="days")
plot!(pβtsmth2[simstart:simend], label = "Model β(t)")

pβtest = interpolate((xt[1:151],),vec(Q_parameter), Gridded(Constant()));

pptest = (pβtest,p_original2)
datasizetest = 151;
tspantest = (0, 150)
datasize = 150;
trange = range(tspantest[1],tspantest[2],length=datasizetest)
probsirtest = ODEProblem(sirmodel1!,u02,tspantest,pptest,saveat=trange)
compartment_soltest = solve(probsirtest,Tsit5())
plot(compartment_soltest[2,1:150],label="Predicted I")
bar!(compartment_sol[2,1:150],label="Target I",color="Grey",alpha=0.3)
