using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

// Uses documented Windows SetupAPI/NewDev APIs to restore one missing device.
// Reference: Microsoft Windows-driver-samples/setup/devcon/cmds.cpp, cmdInstall.
public static class RestoreHyperVStorage
{
    private const string HardwareId = @"root\storvsp";
    private const string InstanceId = @"ROOT\STORVSP\0000";

    [StructLayout(LayoutKind.Sequential)]
    public struct DeviceInfo
    {
        public uint Size;
        public Guid ClassGuid;
        public uint DevInst;
        public UIntPtr Reserved;
    }

    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetupDiGetINFClassW(string file, out Guid classGuid, StringBuilder className, uint size, out uint required);

    [DllImport("setupapi.dll", SetLastError = true, ExactSpelling = true)]
    private static extern IntPtr SetupDiCreateDeviceInfoList(ref Guid classGuid, IntPtr parent);

    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetupDiCreateDeviceInfoW(IntPtr set, string name, ref Guid classGuid, string description, IntPtr parent, uint flags, ref DeviceInfo info);

    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetupDiSetDeviceRegistryPropertyW(IntPtr set, ref DeviceInfo info, uint property, byte[] data, uint size);

    [DllImport("setupapi.dll", SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetupDiCallClassInstaller(uint function, IntPtr set, ref DeviceInfo info);

    [DllImport("setupapi.dll", SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetupDiDestroyDeviceInfoList(IntPtr set);

    [DllImport("newdev.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool UpdateDriverForPlugAndPlayDevicesW(IntPtr parent, string hardwareId, string infPath, uint flags, [MarshalAs(UnmanagedType.Bool)] out bool reboot);

    private static void Check(bool success, string operation)
    {
        if (!success)
        {
            int code = Marshal.GetLastWin32Error();
            throw new Win32Exception(code, operation + " failed (" + code + "): " + new Win32Exception(code).Message);
        }
    }

    // Read-only validation; also used by the compilation check.
    public static Guid ReadInfClass(string infPath)
    {
        Guid classGuid;
        uint required;
        Check(SetupDiGetINFClassW(infPath, out classGuid, new StringBuilder(256), 256, out required), "Read INF class");
        if (classGuid != new Guid("4d36e97d-e325-11ce-bfc1-08002be10318"))
            throw new InvalidOperationException("The INF is not a Windows System-class driver.");
        return classGuid;
    }

    public static bool Restore(string infPath, bool createMissingDevice)
    {
        if (IntPtr.Size != 8) throw new InvalidOperationException("Use 64-bit PowerShell.");
        Guid classGuid = ReadInfClass(infPath);
        if (createMissingDevice)
        {
            IntPtr set = SetupDiCreateDeviceInfoList(ref classGuid, IntPtr.Zero);
            if (set == new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Create device information list failed.");
            try
            {
                DeviceInfo info = new DeviceInfo();
                info.Size = (uint)Marshal.SizeOf(typeof(DeviceInfo));
                // An explicit instance ID prevents duplicate devices on repeated runs.
                Check(SetupDiCreateDeviceInfoW(set, InstanceId, ref classGuid, "Microsoft Hyper-V Virtual Disk Server", IntPtr.Zero, 0, ref info), "Create missing storage device");
                byte[] hardwareIds = Encoding.Unicode.GetBytes(HardwareId + "\0\0");
                Check(SetupDiSetDeviceRegistryPropertyW(set, ref info, 1, hardwareIds, (uint)hardwareIds.Length), "Set storage hardware ID");
                Check(SetupDiCallClassInstaller(0x19, set, ref info), "Register storage device");
            }
            finally { SetupDiDestroyDeviceInfoList(set); }
        }
        bool needsReboot;
        Check(UpdateDriverForPlugAndPlayDevicesW(IntPtr.Zero, HardwareId, infPath, 1, out needsReboot), "Install existing Microsoft storage driver");
        return needsReboot;
    }
}
