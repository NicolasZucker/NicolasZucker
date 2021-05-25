import hoomd
import hoomd.md
import numpy
import math
from numpy.core.numeric import True_
from numpy.lib.shape_base import expand_dims
import matplotlib

def creaangles(N):
    OR=numpy.ndarray(shape=(N,4), dtype=float);
    OR[:,:]=2;
    for i in range(N):
        normmm=numpy.linalg.norm(OR[i,:]);
        while normmm> 1:
            OR[i,:]= [  ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0)]
            normmm=numpy.linalg.norm(OR[i,:]);
    return(OR)


hoomd.context.initialize(""); 


N=50; #nombre de partciules
L=100;

snap = hoomd.data.make_snapshot(N, box=hoomd.data.boxdim(L=100));
numpy.random.seed(12);


my_position= (numpy.random.random((N,3)) * 2 - 1)*(L/2); #10 = taille de la box/2 ici
snap.particles.position[:] = my_position[:];
snap.particles.orientation[:] = creaangles(N)[:];
snap.particles.moment_inertia[:]=[1,1,1];

#print(snap.particles.orientation) #ça marche pas ??


hoomd.init.read_snapshot(snap);

nl = hoomd.md.nlist.cell()

lj = hoomd.md.pair.lj(r_cut=10.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=3.0, sigma=2)
dip = hoomd.md.pair.dipole(r_cut=5, nlist=nl, name=None)
dip.pair_coeff.set('A', 'A', mu=10.0, A=5.0, kappa=0)

hoomd.md.integrate.mode_standard(dt=0.001,aniso=1);

all = hoomd.group.all();

hoomd.md.integrate.brownian(group=all, kT=0.02, seed=123);

hoomd.analyze.log(filename="log-output.log",quantities=['potential_energy', 'temperature'],period=100,overwrite=True);

hoomd.dump.gsd("trajectory.gsd", period=2e2, group=all, overwrite=True);

hoomd.run(1e6);

# Analysis 
import numpy
from matplotlib import pyplot

data = numpy.genfromtxt(fname='log-output.log', skip_header=True);

pyplot.figure(figsize=(4,2.2), dpi=140);
pyplot.plot(data[:,0], data[:,1]);
pyplot.xlabel('time step');
pyplot.ylabel('potential_energy');



pyplot.figure(figsize=(4,2.2), dpi=140);
pyplot.plot(data[:,0], data[:,2]);
pyplot.xlabel('time step');
pyplot.ylabel('temperature');

pyplot.show()