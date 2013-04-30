/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.openpixi.pixi.physics.solver;


import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import example.maven.lastone.DemoKernel;
import org.bridj.Pointer;
import java.nio.ByteOrder;
import static org.bridj.Pointer.*;
import java.io.IOException;
import java.util.ArrayList;
import org.openpixi.pixi.physics.Particle;
import org.openpixi.pixi.physics.Settings;
import org.openpixi.pixi.physics.Simulation;
import org.openpixi.pixi.physics.force.Force;
import org.openpixi.pixi.physics.force.SimpleGridForce;
import static org.openpixi.pixi.physics.solver.BorisCL.inParticles;
import static org.openpixi.pixi.ui.MainBatch.s;


public class BorisCL {
    static ArrayList<Particle> particles;
    static Force f;
    static Pointer<Double> inParticles;
    
    //Particle + Force size(no of fields)
    static int PF_SIZE = 26;
    
    //Number of simulated steps
    static int steps = 500;
    
    /*
     * Converts an array of Particle objects into an array
     * of Doubles so it can be passed to the OpenCL kernel
     */
    static void clConversion(int n, ByteOrder byteOrder){
       
        inParticles = allocateDoubles(n).order(byteOrder);
        
        int k = 0;
        for(int i = 0; i < n; i += PF_SIZE){
            inParticles.set(i + 0, particles.get(k).getX());
            inParticles.set(i + 1, particles.get(k).getY());
            inParticles.set(i + 2, particles.get(k).getRadius());
            inParticles.set(i + 3, particles.get(k).getVx());
            inParticles.set(i + 4, particles.get(k).getVy());
            inParticles.set(i + 5, particles.get(k).getAx());
            inParticles.set(i + 6, particles.get(k).getAy());
            inParticles.set(i + 7, particles.get(k).getMass());
            inParticles.set(i + 8, particles.get(k).getCharge());
            inParticles.set(i + 9, particles.get(k).getPrevX());
            inParticles.set(i + 10, particles.get(k).getPrevY());
            inParticles.set(i + 11, particles.get(k).getChargedensity());
            inParticles.set(i + 12, particles.get(k).getEx());
            inParticles.set(i + 13, particles.get(k).getEy());
            inParticles.set(i + 14, particles.get(k).getBz());
            inParticles.set(i + 15, particles.get(k).getPrevPositionComponentForceX());
            inParticles.set(i + 16, particles.get(k).getPrevPositionComponentForceY());
            inParticles.set(i + 17, particles.get(k).getPrevTangentVelocityComponentOfForceX());
            inParticles.set(i + 18, particles.get(k).getPrevTangentVelocityComponentOfForceY());
            inParticles.set(i + 19, particles.get(k).getPrevBz());
            inParticles.set(i + 20, particles.get(k).getPrevLinearDragCoefficient());
            inParticles.set(i + 21, f.getBz(particles.get(k)));
            inParticles.set(i + 22, f.getPositionComponentofForceX(particles.get(k)));
            inParticles.set(i + 23, f.getPositionComponentofForceY(particles.get(k)));
            inParticles.set(i + 24, f.getTangentVelocityComponentOfForceX(particles.get(k)));
            inParticles.set(i + 25, f.getTangentVelocityComponentOfForceY(particles.get(k++)));
        }
   
    }
    
    public static void main(String[] args) throws IOException {
        s = new Simulation(new Settings());
        Boris b = new Boris();
        int n = s.particles.size() * PF_SIZE;
        int step = 2;
        long t1, t2;
        
        System.out.println("--------------ORIGINAL VERSION---------------");
        t1 = System.currentTimeMillis();
        for(int j = 0; j < s.particles.size(); j++){
            for (int i = 0; i < steps; i++) {
                b.step(s.particles.get(j), new SimpleGridForce(), 0.5);
            }
        }
        t2 = System.currentTimeMillis();
        
        for (int i=0; i < 10; i++) {
            System.out.println("out[" + i + "] = " + s.particles.get(i).getX());
	}
        System.out.println("Elapsed time: " + (t2-t1) + "ms");
        
        
        System.out.println("\n------------PARALLEL VERSION----------------");
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        particles = s.particles;
        f = new SimpleGridForce();
        clConversion(n, byteOrder);
        
        // Create an OpenCL input buffer :
        CLBuffer<Double> inPar = context.createDoubleBuffer(Usage.Input, inParticles);
    
        // Create an OpenCL output buffer :
        CLBuffer<Double> outPar = context.createDoubleBuffer(Usage.Output, n);
        
        //call the kernel
        DemoKernel kernels = new DemoKernel(context);
        CLEvent borisEvt = null;
        int[] globalSizes = new int[] { n };
        CLEvent initEvt = kernels.boris_init(queue, inPar, outPar, n, globalSizes, null);
        
        t1 = System.currentTimeMillis();
        for (int i = 0; i < steps; i++) {
            borisEvt = kernels.boris_step(queue, inPar, outPar, n, globalSizes, null, initEvt);
        }
        t2 = System.currentTimeMillis();
        
        //get output
        Pointer<Double> outPtr = outPar.read(queue, borisEvt); 
        
        //print results
        for (int i = 0; i < 10 * PF_SIZE; i += PF_SIZE)
            System.out.println("out[" + i/PF_SIZE + "] = " + outPtr.get(i));
        System.out.println("Elapsed time: " + (t2-t1) + "ms");
        
    }
}
