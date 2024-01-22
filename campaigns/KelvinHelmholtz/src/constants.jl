module PhysicalConstants

using Unitful

const qe = 1.602177e-19u"C"
const mp = 1.672622e-27u"kg"
const μ0 = 1.256637e-6u"m * kg * s^-2 * A^-2"
const ϵ0 = 8.854187e-12u"m^-3 * kg^-1 * s^4 * A^2"
const c_light = 1 / sqrt(μ0 * ϵ0)

end
