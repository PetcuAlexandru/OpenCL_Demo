__kernel void boris_init(__global double* particles, __global double* out, int n)
{    
    int i = get_global_id(0);
    if (i >= n)
        return;
        
    out[i] = particles[i];
}

__kernel void boris_step(__global double* particles, __global double* out, int n)
{
    int PF_SIZE = 26;
    int step = 0.5;
    
    int i = get_global_id(0) * PF_SIZE;
    if (i >= n)
        return;

//    out[i] = particles[i]; 
    double getPositionComponentofForceX = particles[i + 22];
    double getPositionComponentofForceY = particles[i + 23];
    double getBz = particles[i + 21]; 
    double getTangentVelocityComponentOfForceX = particles[i + 24];
    double getTangentVelocityComponentOfForceY = particles[i + 25];
    double getMass = particles[i + 7];

    double vxminus = particles[i + 3] + getPositionComponentofForceX * step / (2.0 * getMass);
    double vyminus = particles[i + 4] + getPositionComponentofForceY * step / (2.0 * getMass);

    double t_z = (particles[i + 8]) * getBz * step / (2.0 * getMass);
    double s_z = 2 * t_z / (1 + t_z * t_z);

    double vxprime = vxminus + vyminus * t_z;
    double vyprime = vyminus - vxminus * t_z;

    double vxplus = vxminus + vyprime * s_z;
    double vyplus = vyminus - vxprime * s_z;

    out[i + 3] = vxplus + getPositionComponentofForceX * step / (2.0 * getMass) + getTangentVelocityComponentOfForceX * step / getMass;
    out[i + 4] = vyplus + getPositionComponentofForceY * step / (2.0 * getMass) + getTangentVelocityComponentOfForceY * step / getMass;

    out[i + 0] += out[i + 3] * step;
    out[i + 1] += out[i + 4] * step;

}