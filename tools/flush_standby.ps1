param([int]$Command = 4)
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$p = New-Object Security.Principal.WindowsPrincipal($id)
Write-Output ("IsAdmin = " + $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))
$src = @'
using System;
using System.Runtime.InteropServices;
public static class MemFlush {
    [DllImport("ntdll.dll")]
    public static extern int NtSetSystemInformation(int InfoClass, IntPtr Info, int Length);
    public static int Empty(int cmd) {
        IntPtr p = Marshal.AllocHGlobal(4);
        Marshal.WriteInt32(p, cmd);
        int rc = NtSetSystemInformation(0x50, p, 4);
        Marshal.FreeHGlobal(p);
        return rc;
    }
}
'@
try {
    Add-Type -TypeDefinition $src -ErrorAction Stop
    $rc = [MemFlush]::Empty($Command)
    Write-Output ("flush rc = " + $rc + " (0 = success, 0xC0000061 = privilege not held)")
} catch {
    Write-Output ("flush failed: " + $_.Exception.Message)
}
