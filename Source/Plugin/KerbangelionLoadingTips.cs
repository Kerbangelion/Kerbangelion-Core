// Kerbangelion: replaces (or extends) the loading-screen tips with the lines in
// GameData/Kerbangelion/PluginData/LoadingTips.cfg. Edit that file; no recompiling needed.
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Kerbangelion
{
    [KSPAddon(KSPAddon.Startup.Instantly, true)]
    public class LoadingTips : MonoBehaviour
    {
        int frames;
        void Awake() { DontDestroyOnLoad(this); }
        void Update()
        {
            LoadingScreen ls = LoadingScreen.Instance;
            if (ls == null || ls.Screens == null || ls.Screens.Count == 0)
            {
                if (++frames > 600) Destroy(this);
                return;
            }
            try
            {
                string path = Path.Combine(KSPUtil.ApplicationRootPath, "GameData/Kerbangelion/PluginData/LoadingTips.cfg");
                ConfigNode root = File.Exists(path) ? ConfigNode.Load(path) : null;
                ConfigNode n = root == null ? null : (root.HasNode("KERBANGELION_LOADING_TIPS") ? root.GetNode("KERBANGELION_LOADING_TIPS") : root);
                string[] tips = n == null ? new string[0] : n.GetValues("tip");
                bool append = n != null && n.GetValue("mode") == "append";
                if (tips.Length > 0)
                    foreach (LoadingScreen.LoadingScreenState s in ls.Screens)
                    {
                        if (s.tips == null || s.tips.Length == 0) continue;
                        List<string> l = append ? new List<string>(s.tips) : new List<string>();
                        l.AddRange(tips);
                        s.tips = l.ToArray();
                    }
                Debug.Log("[Kerbangelion] loading tips: " + tips.Length + (append ? " added" : " (replacing stock tips)"));
            }
            catch (System.Exception e) { Debug.LogWarning("[Kerbangelion] loading tips failed: " + e.Message); }
            Destroy(this);
        }
    }
}
